"""
============================================================================
 ÉTUDE COMPARATIVE — Méthodes de communication de covariance inter-drones
============================================================================
 On compare 4 façons de transmettre l'incertitude de position d'un voisin
 lors d'une fusion CI sur une mesure de distance scalaire.

 MÉTHODES
 --------
 M0 — Bloc 2x2 exact   : transmet Pⱼ[0:2,0:2]  -> coût 5 floats, référence exacte
 M1 — Scalaire isotrope : transmet trace(Pⱼ)/2  -> coût 3 floats, ignore anisotropie
 M2 — Variance projetée : transmet Hⱼ·Pⱼ·Hⱼᵀ  -> coût 3 floats, exact dans la direction du lien
 M3 — Variance maximale : transmet λmax(Pⱼ)     -> coût 3 floats, borne supérieure garantie

 MÉTRIQUES
 ---------
 MSE        : erreur quadratique position (m²)   — qualité de l'estimation
 NCI        : Normalized Covariance Indicator    — cohérence du filtre
              NCI = (1/T) Σ eₜᵀ Pₜ⁻¹ eₜ  avec e = erreur position (2D)
              NCI ≈ 2 : cohérent   |  NCI << 2 : sur-confiant (dangereux)
              NCI >> 2 : conservateur (sûr mais pessimiste)
 Trace(P)   : somme des variances position       — taille des couloirs σ
 Coût radio : floats émis par lien et par pas    — bande passante

 Tous les résultats sont moyennés sur N_MC runs Monte Carlo (même trajectoire
 vraie, seeds différents) pour séparer la variance stochastique des effets
 structurels de chaque méthode.
============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import inv, eigvalsh
from scipy.linalg import block_diag
from scipy.optimize import minimize_scalar

# ============================================================================
# 1. PARAMÈTRES  (repris à l'identique du filtre distribué)
# ============================================================================
t_max      = 16.0
dt         = 0.1
dt_capteur = 0.5
dt_imu     = 0.1
N_DRONES   = 3
N_VAR      = 8
N          = N_DRONES * N_VAR
n_steps    = int(round(t_max / dt))
ratio_imu  = int(round(dt_imu / dt))
ratio_gps  = int(round(dt_capteur / dt))

sigma_w_accel = 5e-2
sigma_w_autre = 1e-6
sigma_gps  = 0.5
sigma_acc  = 0.01
sigma_d    = 0.5
sigma_R_gps = 0.5
sigma_R_acc = 0.1
sigma_R_d   = 0.5
sigma_P_x, sigma_P_v, sigma_P_a, sigma_P_b = 2.0, 0.5, 0.5, 1.0

# ============================================================================
# 2. MATRICES DU MODÈLE
# ============================================================================
def Fmat(accel_constante):
    a = 1.0 if accel_constante else 0.0
    return np.array([
        [1, 0, dt, 0, 0.5*dt**2, 0,         0, 0],
        [0, 1, 0,  dt, 0,        0.5*dt**2, 0, 0],
        [0, 0, 1,  0,  dt,       0,          0, 0],
        [0, 0, 0,  1,  0,        dt,         0, 0],
        [0, 0, 0,  0,  a,        0,          0, 0],
        [0, 0, 0,  0,  0,        a,          0, 0],
        [0, 0, 0,  0,  0,        0,          1, 0],
        [0, 0, 0,  0,  0,        0,          0, 1]], dtype=float)

def Bmat_global(cols):
    M = np.zeros((8, 6)); M[4, cols[0]] = 1.0; M[5, cols[1]] = 1.0
    return M

def Bmat_local():
    M = np.zeros((8, 2)); M[4, 0] = 1.0; M[5, 1] = 1.0
    return M

F_vrai = block_diag(Fmat(False), Fmat(False), Fmat(False))
B_vrai = np.concatenate((Bmat_global([0,1]), Bmat_global([2,3]), Bmat_global([4,5])), axis=0)

F_loc  = {1: Fmat(False), 2: Fmat(True),  3: Fmat(False)}
B_loc  = {1: Bmat_local(), 2: np.zeros((8,2)), 3: Bmat_local()}

w_sigma = np.full(N, sigma_w_autre)
for b in (0, 8, 16):
    w_sigma[b+4] = w_sigma[b+5] = sigma_w_accel

X_vrai_init = np.concatenate(([0, 10, 1, 0, 0, 0,  0,    0],
                               [10, 0, 1, 0, 0, 0,  0.5, -0.2],
                               [0, -10, 1, 0, 0, 0,  0,    0])).astype(float)

I8 = np.eye(8)

def make_Q_local(drone_id):
    Q = np.eye(8) * 1e-3
    Q[4,4] = Q[5,5] = 0.5**2
    Q[6,6] = Q[7,7] = 1e-5**2
    if drone_id == 2:
        Q[2,2] = Q[3,3] = 0.1**2
        Q[4,4] = Q[5,5] = 1e-2**2
    return Q

def make_P_local():
    P = np.eye(8)
    P[0,0] = P[1,1] = sigma_P_x**2
    P[2,2] = P[3,3] = sigma_P_v**2
    P[4,4] = P[5,5] = sigma_P_a**2
    P[6,6] = P[7,7] = sigma_P_b**2
    return P

# ============================================================================
# 3. MÉTHODES DE CONSTRUCTION DU PAQUET RADIO
#    Chaque méthode définit :
#      - build_paquet(drone)  -> dict émis par le voisin j
#      - n_floats             -> coût en floats (hors position, toujours 2)
#      - var_voisin(paquet, Hj) -> scalaire utilisé dans R augmenté
# ============================================================================

class MethodeBloc2x2:
    """M0 — Bloc exact 2x2 (référence)."""
    nom   = "M0 — Bloc 2×2 exact"
    label = "M0"
    color = "black"
    ls    = "-"
    # Position (2) + 3 coefficients du bloc symétrique 2x2 (Pxx, Pxy, Pyy)
    n_floats_cov = 3

    @staticmethod
    def build_paquet(drone):
        return {"pos": drone.x[0:2].copy(), "data": drone.P[0:2, 0:2].copy()}

    @staticmethod
    def var_voisin(paquet, Hj):
        return float(Hj @ paquet["data"] @ Hj.T)


class MethodeIsotrope:
    """M1 — Scalaire isotrope : trace(P)/2."""
    nom   = "M1 — Variance isotrope (trace/2)"
    label = "M1"
    color = "royalblue"
    ls    = "--"
    n_floats_cov = 1   # un seul scalaire

    @staticmethod
    def build_paquet(drone):
        Ppos = drone.P[0:2, 0:2]
        return {"pos": drone.x[0:2].copy(), "data": np.trace(Ppos) / 2.0}

    @staticmethod
    def var_voisin(paquet, Hj):
        # Hj est unitaire (norme = 1) donc Hj·σ²I·HjT = σ²
        return float(paquet["data"])


class MethodeProjetee:
    """M2 — Variance projetée : Hj · P · HjT calculée par le voisin."""
    nom   = "M2 — Variance projetée (Hj·P·Hj)"
    label = "M2"
    color = "darkorange"
    ls    = "-."
    n_floats_cov = 1

    @staticmethod
    def build_paquet(drone):
        # Le voisin j calcule lui-même sa contribution scalaire dans chaque
        # direction de lien possible. On stocke le bloc 2x2 et on projette
        # au moment de l'usage (c'est équivalent : le voisin projette avec
        # sa propre estimation de Hj avant émission dans un vrai système).
        # Ici on passe le bloc 2x2 et on projette côté récepteur avec le Hj
        # que lui calcule depuis sa propre position — c'est l'approximation
        # "géométrie localement connue".
        return {"pos": drone.x[0:2].copy(), "data": drone.P[0:2, 0:2].copy()}

    @staticmethod
    def var_voisin(paquet, Hj):
        # Même calcul que M0 — dans un vrai système embarqué, le voisin
        # calculerait Hj = f(pos_i - pos_j) depuis sa propre pos et l'émettrait
        # comme scalaire. Ici l'approximation est que les deux drones
        # estiment la même géométrie (petites erreurs de position).
        return float(Hj @ paquet["data"] @ Hj.T)


class MethodeMax:
    """M3 — Variance maximale : λmax(P[0:2, 0:2])."""
    nom   = "M3 — Variance maximale (λmax)"
    label = "M3"
    color = "crimson"
    ls    = ":"
    n_floats_cov = 1

    @staticmethod
    def build_paquet(drone):
        Ppos = drone.P[0:2, 0:2]
        return {"pos": drone.x[0:2].copy(), "data": float(eigvalsh(Ppos).max())}

    @staticmethod
    def var_voisin(paquet, Hj):
        return float(paquet["data"])


METHODES = [MethodeBloc2x2, MethodeIsotrope, MethodeProjetee, MethodeMax]

# Coût radio total par pas capteur : 3 drones × (2 pos + n_floats_cov)
def cout_floats(methode):
    return 3 * (2 + methode.n_floats_cov)

# ============================================================================
# 4. CLASSE ESTIMATEUR DISTRIBUÉ (généralisée : accepte la méthode en paramètre)
# ============================================================================
class DroneDistribue:
    def __init__(self, drone_id, x_init, P_init, Q_local):
        self.id  = drone_id
        self.x   = x_init.copy()
        self.P   = P_init.copy()
        self.Q   = Q_local.copy()
        self.F   = F_loc[drone_id]
        self.B   = B_loc[drone_id]

    def predict(self, u):
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_local(self, mes, H, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ inv(S)
        innov = mes - H @ self.x
        self.x = self.x + K @ innov
        A = I8 - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T

    def update_CI_batch(self, liens, R_scalaire, methode, diagnostics=None):
        """
        Fusion CI batch avec la méthode de communication passée en paramètre.
        liens : liste de (d_mesure, paquet_voisin)
        diagnostics : si fourni (liste), on y ajoute pour chaque lien un tuple
                      (angle_lien_deg, var_utilisee, var_exacte) pour les
                      métriques 'erreur de variance vs angle'. N'altère PAS
                      le calcul du filtre.
        """
        if not liens:
            return
        rows_Hi, innovs, R_diag = [], [], []

        for d_mesure, paquet in liens:
            xi, yi = self.x[0], self.x[1]
            xj, yj = paquet["pos"]
            d_pred = np.hypot(xi - xj, yi - yj)
            if d_pred < 1e-4:
                d_pred = 1e-4

            Hi      = np.zeros(8)
            Hi[0]   = (xi - xj) / d_pred
            Hi[1]   = (yi - yj) / d_pred
            Hj      = np.array([-(xi - xj) / d_pred, -(yi - yj) / d_pred])

            # C'est ici que les méthodes divergent :
            var_j = methode.var_voisin(paquet, Hj)

            # --- Diagnostic : angle du lien + écart à la variance exacte -----
            if diagnostics is not None and "Ppos_exact" in paquet:
                angle = np.degrees(np.arctan2(Hj[1], Hj[0])) % 360.0
                var_exacte = float(Hj @ paquet["Ppos_exact"] @ Hj.T)
                diagnostics.append((angle, var_j, var_exacte))

            rows_Hi.append(Hi)
            innovs.append(d_mesure - d_pred)
            R_diag.append(R_scalaire + var_j)

        H     = np.vstack(rows_Hi)
        innov = np.array(innovs)
        Rv    = np.diag(R_diag)

        def cout_CI(omega):
            Pi  = self.P / omega
            S   = H @ Pi @ H.T + Rv / (1.0 - omega)
            K   = Pi @ H.T @ inv(S)
            return np.trace(Pi - K @ H @ Pi)

        sol   = minimize_scalar(cout_CI, bounds=(0.01, 0.99), method='bounded')
        omega = sol.x
        Pi    = self.P / omega
        S     = H @ Pi @ H.T + Rv / (1.0 - omega)
        K     = Pi @ H.T @ inv(S)
        self.x = self.x + K @ innov
        self.P = Pi - K @ H @ Pi

# ============================================================================
# 5. SIMULATION MONO-RUN POUR UNE MÉTHODE DONNÉE
# ============================================================================
def run_une_methode(methode, seed):
    np.random.seed(seed)

    X_vrai = X_vrai_init.copy()

    erreur_init = np.zeros(N)
    for b in (0, 8, 16):
        erreur_init[b:b+2] = np.random.normal(0, 2.0, size=2)
    X_est0 = X_vrai + erreur_init

    drones = {
        i: DroneDistribue(i, X_est0[(i-1)*8:i*8], make_P_local(), make_Q_local(i))
        for i in (1, 2, 3)
    }

    # Tableaux de résultats
    err_pos  = np.zeros((n_steps+1, N_DRONES, 2))   # erreur position (x,y) par drone
    P_pos_tr = np.zeros((n_steps+1, N_DRONES))       # trace P position par drone
    P_pos_m  = np.zeros((n_steps+1, N_DRONES, 2, 2)) # matrice P position par drone

    # Diagnostic variance/angle : liste de (angle_deg, var_utilisee, var_exacte)
    diag_var = []

    phi_x = phi_y = 0.0

    for k in range(1, n_steps+1):
        step = k - 1
        t    = k * dt

        # Commande
        if step < n_steps / 3:
            u_vrai = np.array([1., 0., 1., 0., 1., 0.])
        elif step < 2 * n_steps / 3:
            phi_x += 5 * dt; phi_y += 1 * dt
            u_vrai = np.array([np.cos(phi_x), np.sin(phi_y),
                               np.cos(phi_x), np.sin(phi_y),
                               np.cos(phi_x), np.sin(phi_y)])
        else:
            u_vrai = np.array([1., 0., 1., 0., 1., 0.])

        err_cmd = np.random.normal(0, 0.1, size=6); err_cmd[0:2] = 0.0
        u_kalman = u_vrai + err_cmd

        X_vrai = F_vrai @ X_vrai + B_vrai @ u_vrai + np.random.normal(0, 1, N) * w_sigma

        drones[1].predict(u_kalman[0:2])
        drones[2].predict(np.zeros(2))
        drones[3].predict(u_kalman[4:6])

        # IMU drone 2
        if step % ratio_imu == 0:
            mes = np.array([X_vrai[12] + X_vrai[14] + np.random.normal(0, sigma_acc),
                            X_vrai[13] + X_vrai[15] + np.random.normal(0, sigma_acc)])
            H = np.zeros((2, 8)); H[0,4]=H[0,6]=1.0; H[1,5]=H[1,7]=1.0
            drones[2].update_local(mes, H, np.diag([sigma_R_acc**2]*2))

        # GPS drone 1 + distances
        if step % ratio_gps == 0:
            z = np.array([X_vrai[0] + np.random.normal(0, sigma_gps),
                          X_vrai[1] + np.random.normal(0, sigma_gps)])
            H = np.zeros((2, 8)); H[0,0]=1.0; H[1,1]=1.0
            drones[1].update_local(z, H, np.diag([sigma_R_gps**2]*2))

            d12v = np.hypot(X_vrai[0]-X_vrai[8],  X_vrai[1]-X_vrai[9])
            d23v = np.hypot(X_vrai[8]-X_vrai[16], X_vrai[9]-X_vrai[17])
            d13v = np.hypot(X_vrai[0]-X_vrai[16], X_vrai[1]-X_vrai[17])
            z12  = d12v + np.random.normal(0, sigma_d)
            z23  = d23v + np.random.normal(0, sigma_d)
            z13  = d13v + np.random.normal(0, sigma_d)

            paquets = {i: methode.build_paquet(drones[i]) for i in (1,2,3)}
            # Injecte le vrai bloc 2x2 position pour le calcul de la variance
            # exacte (diagnostic uniquement, n'altère pas le filtre).
            for i in (1, 2, 3):
                paquets[i]["Ppos_exact"] = drones[i].P[0:2, 0:2].copy()

            drones[1].update_CI_batch([(z12, paquets[2]), (z13, paquets[3])],
                                       sigma_R_d**2, methode, diagnostics=diag_var)
            drones[2].update_CI_batch([(z12, paquets[1]), (z23, paquets[3])],
                                       sigma_R_d**2, methode, diagnostics=diag_var)
            drones[3].update_CI_batch([(z23, paquets[2]), (z13, paquets[1])],
                                       sigma_R_d**2, methode, diagnostics=diag_var)

        # Enregistrement
        for j, (d, b) in enumerate([(1,0),(2,8),(3,16)]):
            err_pos[k, j]   = drones[d].x[0:2] - X_vrai[b:b+2]
            P_pos_tr[k, j]  = drones[d].P[0,0] + drones[d].P[1,1]
            P_pos_m[k, j]   = drones[d].P[0:2, 0:2]

    return err_pos, P_pos_tr, P_pos_m, np.array(diag_var) if diag_var else np.empty((0,3))

# ============================================================================
# 6. CALCUL DES MÉTRIQUES SUR N_MC RUNS
# ============================================================================
# Seuil de couverture : ellipse à k_sigma. En 2D, P(||e||² < k²·... ) :
#   le test "e^T P^-1 e < seuil" suit un chi² à 2 ddl.
#   Seuil 3sigma <-> chi2(2) à 99.7%? On utilise le quantile chi² à 2 ddl.
from scipy.stats import chi2
SEUIL_3SIGMA = chi2.ppf(0.997, df=2)   # ~11.6 : couverture cible 99.7%

def calcule_metriques(methode, N_MC=30, seeds=None):
    """
    Retourne un dictionnaire de métriques (familles 1 à 4) :
      mse        : (N_MC, 3)        — MSE position par run/drone           [F1]
      rmse_t     : (T, 3)           — RMSE temporel moyen sur les runs     [F1]
      nci        : (N_MC, 3)        — NCI moyen temporel par run/drone     [F2]
      nci_min    : (3,)             — NCI instantané minimum (pire cas)    [F2]
      couverture : (3,)             — taux de présence vraie pos dans 3σ   [F2]
      trace_mean : (T, 3)           — trace(P) moyenne                     —
      trace_std  : (T, 3)           — écart-type trace(P)                  —
      diag_var   : (M, 3)           — (angle, var_util, var_exacte) cumulé [F3]
      var_err_var: scalaire         — variance de l'écart var_util-exacte  [F3]
    """
    if seeds is None:
        seeds = range(N_MC)

    all_mse    = np.zeros((N_MC, N_DRONES))
    all_nci    = np.zeros((N_MC, N_DRONES))
    all_trace  = np.zeros((N_MC, n_steps+1, N_DRONES))
    all_rmse_t = np.zeros((N_MC, n_steps+1, N_DRONES))
    nci_min    = np.full(N_DRONES, np.inf)
    n_dans_3s  = np.zeros(N_DRONES)   # comptage couverture
    n_total    = np.zeros(N_DRONES)
    diag_all   = []

    for run, seed in enumerate(seeds):
        err_pos, P_pos_tr, P_pos_m, diag_var = run_une_methode(methode, seed)
        if diag_var.size:
            diag_all.append(diag_var)

        for j in range(N_DRONES):
            # [F1] MSE : moyenne temporelle de ||erreur||²
            all_mse[run, j] = np.mean(np.sum(err_pos[1:, j, :]**2, axis=1))

            # [F1] RMSE temporel : ||erreur|| à chaque instant
            all_rmse_t[run, :, j] = np.sqrt(np.sum(err_pos[:, j, :]**2, axis=1))

            # [F2] NCI instantané + couverture
            nci_vals = []
            for k in range(1, n_steps+1):
                e   = err_pos[k, j]
                Pij = P_pos_m[k, j]
                try:
                    d2 = float(e @ inv(Pij) @ e)   # distance de Mahalanobis²
                except Exception:
                    continue
                nci_vals.append(d2)
                nci_min[j] = min(nci_min[j], d2)
                n_total[j]   += 1
                if d2 <= SEUIL_3SIGMA:
                    n_dans_3s[j] += 1
            all_nci[run, j] = np.mean(nci_vals) if nci_vals else np.nan
            all_trace[run, :, j] = P_pos_tr[:, j]

    trace_mean = np.mean(all_trace, axis=0)
    trace_std  = np.std(all_trace,  axis=0)
    rmse_t     = np.mean(all_rmse_t, axis=0)
    couverture = n_dans_3s / np.maximum(n_total, 1)

    diag_var = np.vstack(diag_all) if diag_all else np.empty((0, 3))
    # [F3] variance de l'écart entre variance utilisée et variance exacte
    if diag_var.size:
        ecart = diag_var[:, 1] - diag_var[:, 2]
        var_err_var = float(np.var(ecart))
    else:
        var_err_var = 0.0

    return {
        "mse"        : all_mse,
        "rmse_t"     : rmse_t,
        "nci"        : all_nci,
        "nci_min"    : nci_min,
        "couverture" : couverture,
        "trace_mean" : trace_mean,
        "trace_std"  : trace_std,
        "diag_var"   : diag_var,
        "var_err_var": var_err_var,
    }

# ============================================================================
# 7. CALCUL DE TOUTES LES MÉTHODES
# ============================================================================
N_MC  = 40
seeds = list(range(N_MC))
temps = np.arange(n_steps+1) * dt

print(f"Étude sur {N_MC} runs Monte Carlo, {n_steps} pas de simulation chacun.")
print("─" * 60)

resultats = {}
for m in METHODES:
    print(f"  Calcul {m.nom} ...", flush=True)
    R = calcule_metriques(m, N_MC=N_MC, seeds=seeds)
    R["methode"] = m
    R["cout"]    = cout_floats(m)
    resultats[m.label] = R

# ============================================================================
# 8. TABLEAU RÉCAPITULATIF EN CONSOLE  (vue 3 drones, toutes familles)
# ============================================================================
def moy3(r, cle):
    """Moyenne sur les 3 drones d'une métrique (N_MC,3) ou (3,)."""
    arr = r[cle]
    if arr.ndim == 2:
        return arr.mean()
    return arr.mean()

print("\n" + "═"*92)
print(f"{'Méthode':<30} {'Coût':>5} {'MSE':>8} {'NCI moy':>9} {'NCI min':>9} "
      f"{'Couv.3σ':>9} {'Var(err)':>10}")
print(f"{'':30} {'flt':>5} {'(m²)':>8} {'(≈2)':>9} {'(pire)':>9} "
      f"{'(≈99.7%)':>9} {'var (F3)':>10}")
print("─"*92)
for label, r in resultats.items():
    m = r["methode"]
    mse_m  = r["mse"].mean()
    nci_m  = r["nci"].mean()
    nci_mn = r["nci_min"].min()
    couv   = r["couverture"].mean() * 100
    vev    = r["var_err_var"]
    print(f"  {m.nom:<28} {r['cout']:>5d} {mse_m:>8.3f} {nci_m:>9.3f} "
          f"{nci_mn:>9.3f} {couv:>8.1f}% {vev:>10.4f}")
print("═"*92)
print("  Familles : F1 précision (MSE) | F2 cohérence (NCI, couverture) | "
      "F3 géométrie (Var(err))")
print("  NCI ≈ 2 : cohérent | NCI min très bas : sur-confiance ponctuelle | "
      "Couv. < 99% : risque")
print("  Var(err) = 0 : insensible à l'angle (M0/M2) | > 0 : sensible (M1/M3)")

# ── Tableaux détaillés MSE et NCI par drone (à recopier en présentation) ─────
labels_drone = ["Drone 1 (GPS)", "Drone 2 (IMU)", "Drone 3 (sans capteur)"]

print("\n\n┌" + "─"*70 + "┐")
print("│  TABLEAU MSE (m²) — erreur quadratique moyenne de position" + " "*12 + "│")
print("└" + "─"*70 + "┘")
print(f"{'Méthode':<22}" + "".join(f"{d:>17}" for d in labels_drone))
print("-"*73)
for label, r in resultats.items():
    mse_par_drone = r["mse"].mean(axis=0)        # (3,)
    ligne = f"{r['methode'].label + ' — ' + r['methode'].nom.split('—')[1].strip():<22}"
    ligne = f"{r['methode'].label:<6}"
    ligne += "".join(f"{v:>17.3f}" for v in mse_par_drone)
    print(ligne)

print("\n┌" + "─"*70 + "┐")
print("│  TABLEAU NCI — cohérence (cible ≈ 2 ; < 2 conservateur ; > 2 risqué)" + " "*1 + "│")
print("└" + "─"*70 + "┘")
print(f"{'Méthode':<6}" + "".join(f"{d:>17}" for d in labels_drone))
print("-"*73)
for label, r in resultats.items():
    nci_par_drone = r["nci"].mean(axis=0)        # (3,)
    ligne = f"{r['methode'].label:<6}"
    ligne += "".join(f"{v:>17.3f}" for v in nci_par_drone)
    print(ligne)
print()

# ============================================================================
# 9. FIGURES  (2 figures épurées pour la présentation)
#
#   Fig 1 — Erreur de variance vs angle : signature géométrique de chaque méthode
#           (4 méthodes côte à côte, version simplifiée : nuage + médiane)
#   Fig 2 — Classement global : barres horizontales triées du meilleur au pire
#
#   (La synthèse chiffrée MSE / NCI est imprimée dans le terminal ci-dessus.)
# ============================================================================

# ── Figure 1 : Erreur de variance transmise vs angle du lien ────────────────
# Lecture : chaque point est une fusion CI. Y = variance utilisée − exacte.
#   - trait plat sur 0  -> méthode exacte (M0, M2)
#   - nuage qui oscille -> sensible à l'angle (M1)
#   - nuage toujours > 0 -> sur-estime toujours (M3)
# La ligne foncée est la médiane glissante : elle résume la tendance.
# ─────────────────────────────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
fig1.suptitle("Erreur de variance transmise selon l'orientation du lien",
              fontsize=13, fontweight='bold')

# Plage Y commune (on coupe les 1% d'outliers extrêmes pour la lisibilité)
ecarts_tous = np.concatenate([resultats[m.label]["diag_var"][:, 1]
                              - resultats[m.label]["diag_var"][:, 2]
                              for m in METHODES])
y_lim = np.percentile(np.abs(ecarts_tous), 99)

for ax, m in zip(axes1, METHODES):
    dv    = resultats[m.label]["diag_var"]
    angle = dv[:, 0]
    ecart = dv[:, 1] - dv[:, 2]

    # Nuage de points discret
    ax.scatter(angle, ecart, s=3, color=m.color, alpha=0.25)

    # Médiane glissante par tranche de 20° (la tendance qui résume tout)
    bins = np.arange(0, 361, 20)
    mids = (bins[:-1] + bins[1:]) / 2
    meds = [np.median(ecart[(angle >= bins[k]) & (angle < bins[k+1])])
            if np.any((angle >= bins[k]) & (angle < bins[k+1])) else np.nan
            for k in range(len(bins) - 1)]
    ax.plot(mids, meds, color=m.color, lw=2.5, zorder=5)

    ax.axhline(0, color='black', lw=1)
    ax.set_xlim(0, 360); ax.set_xticks([0, 180, 360])
    ax.set_ylim(-y_lim, y_lim)
    ax.set_title(m.label, fontsize=12, color=m.color, fontweight='bold')
    ax.set_xlabel("Angle du lien (°)")
    ax.grid(True, linestyle=':', alpha=0.4)

axes1[0].set_ylabel("Variance utilisée − exacte")
fig1.tight_layout()


# ── Figure 2 : Classement global des méthodes ───────────────────────────────
# On combine les 4 critères en un score unique (0 = meilleur, 1 = pire).
# Chaque critère est normalisé puis moyenné. Barres triées = classement direct.
# ─────────────────────────────────────────────────────────────────────────────
def normalise(vals):
    arr = np.array(vals, dtype=float)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)

# Les 4 critères (tous "plus petit = mieux")
crit_mse  = normalise([resultats[m.label]["mse"].mean()              for m in METHODES])
crit_nci  = normalise([abs(resultats[m.label]["nci"].mean() - 2)     for m in METHODES])
crit_geo  = normalise([resultats[m.label]["var_err_var"]             for m in METHODES])
crit_cout = normalise([resultats[m.label]["cout"]                    for m in METHODES])

# Score global = moyenne des 4 critères normalisés
scores = (crit_mse + crit_nci + crit_geo + crit_cout) / 4.0

# Tri du meilleur (score bas) au pire
ordre = np.argsort(scores)
methodes_triees = [METHODES[i] for i in ordre]
scores_tries    = scores[ordre]

fig2, ax2 = plt.subplots(figsize=(9, 4.5))
y_pos = np.arange(len(methodes_triees))
barres = ax2.barh(y_pos, scores_tries,
                  color=[m.color for m in methodes_triees], alpha=0.85)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([m.nom for m in methodes_triees], fontsize=10)
ax2.invert_yaxis()                       # meilleur en haut
ax2.set_xlabel("Score global  (0 = meilleur, 1 = pire)", fontsize=11)
ax2.set_title("Classement global des méthodes\n"
              "moyenne de 4 critères : précision, cohérence, géométrie, coût radio",
              fontsize=12, fontweight='bold')
ax2.grid(True, axis='x', linestyle=':', alpha=0.5)

# Valeur du score affichée au bout de chaque barre
for bar, sc in zip(barres, scores_tries):
    ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f"{sc:.2f}", va='center', fontsize=9, fontweight='bold')
ax2.set_xlim(0, 1.1)
fig2.tight_layout()

plt.show()
