# -*- coding: utf-8 -*-
"""
EKF DÉCENTRALISÉ — fusion Covariance Intersection (CI)
4 méthodes de communication (M0 bloc 2x2 exact, M1 isotrope, M2 projetée, M3 lambda_max)

Modèle ALIGNÉ sur le centralisé "Scénario D" (biais estimé, GPS+IMU+distances) :
  - mêmes Q, P0, tirage d'erreur initiale, commandes, bruits
  - drone 2 en modèle "accélération constante" (n'observe pas sa commande)

Sorties :
  - MSE de position par méthode et par drone (fenêtre convergée t >= T_CONV)
  - NCI (cohérence), couverture 3-sigma, taux de divergence
  - Figures : 8 variables d'état en colonnes (4x2) + couloir ±3σ + runs MC
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')                      # backend fichier ; retirer pour affichage interactif
import matplotlib.pyplot as plt
from numpy.linalg import inv, eigvalsh
from scipy.linalg import block_diag
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

# =========================================================================
# 1. Paramètres (identiques au centralisé)
# =========================================================================
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

T_CONV   = 4.0                              # début de la fenêtre convergée
IDX_CONV = int(round(T_CONV / dt))

# =========================================================================
# 2. Modèles dynamiques
# =========================================================================
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

# Vérité : les 3 drones ont un modèle à accélération commandée (a=0 dans F, commande via B)
F_vrai = block_diag(Fmat(False), Fmat(False), Fmat(False))
B_vrai = np.concatenate((Bmat_global([0, 1]), Bmat_global([2, 3]), Bmat_global([4, 5])), axis=0)

# Filtres locaux : drone 2 en "accélération constante" (a=1), sans commande (B nul)
F_loc = {1: Fmat(False), 2: Fmat(True),  3: Fmat(False)}
B_loc = {1: Bmat_local(), 2: np.zeros((8, 2)), 3: Bmat_local()}

w_sigma = np.full(N, sigma_w_autre)
for b in (0, 8, 16):
    w_sigma[b+4] = w_sigma[b+5] = sigma_w_accel

X_vrai_init = np.concatenate(([0,  10, 1, 0, 0, 0,  0,    0],
                               [10,  0, 1, 0, 0, 0,  0.5, -0.2],
                               [0, -10, 1, 0, 0, 0,  0,    0])).astype(float)

I8 = np.eye(8)

# =========================================================================
# 3. Q et P0 locaux — ALIGNÉS SUR LE CENTRALISÉ "D"
# =========================================================================
def make_Q_local(drone_id):
    """
    Centralisé D :
       drones 1,3 : Q[accel]=0.5**2 ; Q[biais]=1e-9**2
       drone 2    : Q[vitesse]=0.1**2 ; Q[accel]=1.0 (=1**2) ; Q[biais]=1e-9**2
    """
    Q = np.eye(8) * 1e-3
    Q[4, 4] = Q[5, 5] = 0.5**2          # accélération (drones 1 et 3)
    Q[6, 6] = Q[7, 7] = 1e-9**2         # biais
    if drone_id == 2:
        Q[2, 2] = Q[3, 3] = 0.1**2      # vitesse (comme Q[10,11] centralisé)
        Q[4, 4] = Q[5, 5] = 1.0         # accélération (comme Q[12,13] centralisé)
    return Q

def make_P_local(drone_id=None):
    """
    Centralisé D : P0 diagonale standard, sauf drone 2 dont la vitesse
    est gonflée (P[10,11] = sigma_P_v**2 + 2).
    """
    P = np.eye(8)
    P[0, 0] = P[1, 1] = sigma_P_x**2
    P[2, 2] = P[3, 3] = sigma_P_v**2
    P[4, 4] = P[5, 5] = sigma_P_a**2
    P[6, 6] = P[7, 7] = sigma_P_b**2
    if drone_id == 2:
        P[2, 2] = P[3, 3] = sigma_P_v**2 + 2
    return P

# =========================================================================
# 4. Méthodes de communication (contenu du paquet + variance vue du voisin)
# =========================================================================
class MethodeBloc2x2:
    """M0 — Bloc exact 2x2 (référence)."""
    nom = "M0 - Bloc 2x2 exact"; label = "M0"; color = "black"; ls = "-"; n_floats_cov = 3
    @staticmethod
    def build_paquet(drone):
        return {"pos": drone.x[0:2].copy(), "data": drone.P[0:2, 0:2].copy()}
    @staticmethod
    def var_voisin(paquet, Hj):
        return float(Hj @ paquet["data"] @ Hj.T)

class MethodeIsotrope:
    """M1 — Scalaire isotrope : trace(P)/2."""
    nom = "M1 - Variance isotrope (trace/2)"; label = "M1"; color = "royalblue"; ls = "--"; n_floats_cov = 1
    @staticmethod
    def build_paquet(drone):
        Ppos = drone.P[0:2, 0:2]
        return {"pos": drone.x[0:2].copy(), "data": np.trace(Ppos) / 2.0}
    @staticmethod
    def var_voisin(paquet, Hj):
        return float(paquet["data"])           # Hj unitaire => Hj·σ²I·HjT = σ²

class MethodeProjetee:
    """M2 — Variance projetée : Hj·P·HjT."""
    nom = "M2 - Variance projetee (Hj.P.Hj)"; label = "M2"; color = "darkorange"; ls = "-."; n_floats_cov = 1
    @staticmethod
    def build_paquet(drone):
        return {"pos": drone.x[0:2].copy(), "data": drone.P[0:2, 0:2].copy()}
    @staticmethod
    def var_voisin(paquet, Hj):
        return float(Hj @ paquet["data"] @ Hj.T)

class MethodeMax:
    """M3 — Variance maximale : lambda_max(P[0:2,0:2])."""
    nom = "M3 - Variance maximale (lmax)"; label = "M3"; color = "crimson"; ls = ":"; n_floats_cov = 1
    @staticmethod
    def build_paquet(drone):
        Ppos = drone.P[0:2, 0:2]
        return {"pos": drone.x[0:2].copy(), "data": float(eigvalsh(Ppos).max())}
    @staticmethod
    def var_voisin(paquet, Hj):
        return float(paquet["data"])

METHODES = [MethodeBloc2x2, MethodeIsotrope, MethodeProjetee, MethodeMax]

def cout_floats(methode):
    return 3 * (2 + methode.n_floats_cov)

# =========================================================================
# 5. Drone distribué
# =========================================================================
class DroneDistribue:
    def __init__(self, drone_id, x_init, P_init, Q_local):
        self.id = drone_id
        self.x  = x_init.copy()
        self.P  = P_init.copy()
        self.Q  = Q_local.copy()
        self.F  = F_loc[drone_id]
        self.B  = B_loc[drone_id]

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
        Fusion CI batch. liens : liste de (distance_mesurée, paquet_voisin).
        diagnostics : liste optionnelle pour la métrique variance/angle.
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
            Hi = np.zeros(8)
            Hi[0] = (xi - xj) / d_pred
            Hi[1] = (yi - yj) / d_pred
            Hj = np.array([-(xi - xj) / d_pred, -(yi - yj) / d_pred])
            var_j = methode.var_voisin(paquet, Hj)
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
            Pi = self.P / omega
            S  = H @ Pi @ H.T + Rv / (1.0 - omega)
            K  = Pi @ H.T @ inv(S)
            return np.trace(Pi - K @ H @ Pi)

        sol   = minimize_scalar(cout_CI, bounds=(0.01, 0.99), method='bounded')
        omega = sol.x
        Pi = self.P / omega
        S  = H @ Pi @ H.T + Rv / (1.0 - omega)
        K  = Pi @ H.T @ inv(S)
        self.x = self.x + K @ innov
        self.P = Pi - K @ H @ Pi

# =========================================================================
# 6. Un run complet pour une méthode donnée
# =========================================================================
def run_une_methode(methode, seed):
    np.random.seed(seed)
    X_vrai = X_vrai_init.copy()

    # erreur initiale : MÊME tirage/ordre que le centralisé
    erreur_init = np.zeros(N)
    for b in (0, 8, 16):
        erreur_init[b:b+2]   = np.random.normal(0, sigma_P_x, size=2)
        erreur_init[b+2:b+4] = np.random.normal(0, sigma_P_v, size=2)
        erreur_init[b+4:b+6] = np.random.normal(0, sigma_P_a, size=2)
        erreur_init[b+6:b+8] = np.random.normal(0, sigma_P_b, size=2)
    erreur_init[6:8]   = [0, 0]
    erreur_init[22:24] = [0, 0]
    X_est0 = X_vrai + erreur_init

    drones = {i: DroneDistribue(i, X_est0[(i-1)*8:i*8], make_P_local(i), make_Q_local(i))
              for i in (1, 2, 3)}

    err_pos  = np.zeros((n_steps+1, N_DRONES, 2))
    P_pos_m  = np.zeros((n_steps+1, N_DRONES, 2, 2))
    err8     = np.zeros((n_steps+1, N_DRONES, 8))
    Pdiag8   = np.zeros((n_steps+1, N_DRONES, 8))
    diag_var = []

    # enregistrement t=0
    for j, (d, b) in enumerate([(1, 0), (2, 8), (3, 16)]):
        err_pos[0, j] = drones[d].x[0:2] - X_vrai[b:b+2]
        P_pos_m[0, j] = drones[d].P[0:2, 0:2]
        err8[0, j]    = drones[d].x[0:8] - X_vrai[b:b+8]
        Pdiag8[0, j]  = np.diag(drones[d].P)[0:8]

    phi_x = phi_y = 0.0
    for k in range(1, n_steps+1):
        step = k - 1
        # --- commande vraie ---
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

        # --- propagation vérité ---
        X_vrai = F_vrai @ X_vrai + B_vrai @ u_vrai + np.random.normal(0, 1, N) * w_sigma

        # --- prédiction locale (drone 2 sans commande) ---
        drones[1].predict(u_kalman[0:2])
        drones[2].predict(np.zeros(2))
        drones[3].predict(u_kalman[4:6])

        # --- IMU drone 2 ---
        if step % ratio_imu == 0:
            mes = np.array([X_vrai[12] + X_vrai[14] + np.random.normal(0, sigma_acc),
                            X_vrai[13] + X_vrai[15] + np.random.normal(0, sigma_acc)])
            H = np.zeros((2, 8)); H[0, 4] = H[0, 6] = 1.0; H[1, 5] = H[1, 7] = 1.0
            drones[2].update_local(mes, H, np.diag([sigma_R_acc**2]*2))

        # --- GPS drone 1 + distances (fusion CI) ---
        if step % ratio_gps == 0:
            z = np.array([X_vrai[0] + np.random.normal(0, sigma_gps),
                          X_vrai[1] + np.random.normal(0, sigma_gps)])
            H = np.zeros((2, 8)); H[0, 0] = 1.0; H[1, 1] = 1.0
            drones[1].update_local(z, H, np.diag([sigma_R_gps**2]*2))

            d12v = np.hypot(X_vrai[0]-X_vrai[8],  X_vrai[1]-X_vrai[9])
            d23v = np.hypot(X_vrai[8]-X_vrai[16], X_vrai[9]-X_vrai[17])
            d13v = np.hypot(X_vrai[0]-X_vrai[16], X_vrai[1]-X_vrai[17])
            z12 = d12v + np.random.normal(0, sigma_d)
            z23 = d23v + np.random.normal(0, sigma_d)
            z13 = d13v + np.random.normal(0, sigma_d)

            paquets = {i: methode.build_paquet(drones[i]) for i in (1, 2, 3)}
            for i in (1, 2, 3):
                paquets[i]["Ppos_exact"] = drones[i].P[0:2, 0:2].copy()

            drones[1].update_CI_batch([(z12, paquets[2]), (z13, paquets[3])],
                                       sigma_R_d**2, methode, diagnostics=diag_var)
            drones[2].update_CI_batch([(z12, paquets[1]), (z23, paquets[3])],
                                       sigma_R_d**2, methode, diagnostics=diag_var)
            drones[3].update_CI_batch([(z23, paquets[2]), (z13, paquets[1])],
                                       sigma_R_d**2, methode, diagnostics=diag_var)

        # --- enregistrement ---
        for j, (d, b) in enumerate([(1, 0), (2, 8), (3, 16)]):
            err_pos[k, j] = drones[d].x[0:2] - X_vrai[b:b+2]
            P_pos_m[k, j] = drones[d].P[0:2, 0:2]
            err8[k, j]    = drones[d].x[0:8] - X_vrai[b:b+8]
            Pdiag8[k, j]  = np.diag(drones[d].P)[0:8]

    diag_arr = np.array(diag_var) if diag_var else np.empty((0, 3))
    return err_pos, P_pos_m, diag_arr, err8, Pdiag8

# =========================================================================
# 7. Métriques Monte-Carlo (MSE fenêtrée, NCI, couverture, divergence)
# =========================================================================
SEUIL_3SIGMA = chi2.ppf(0.997, df=2)       # ~11.83 : couverture cible 99.7% (2 DDL)
SEUIL_DIVERG = 50.0                         # MSE (m²) au-delà duquel un run est jugé "divergent"
                                            # (choisi bien au-dessus des médianes ~8-22 m² pour
                                            #  n'attraper que les vrais outliers ; ajuste si besoin)

def calcule_metriques(methode, N_MC, seeds):
    all_mse   = np.zeros((N_MC, N_DRONES))
    all_nci   = np.zeros((N_MC, N_DRONES))
    n_dans_3s = np.zeros(N_DRONES)
    n_total   = np.zeros(N_DRONES)
    err_hist  = np.zeros((N_MC, n_steps+1, N_DRONES, 8))
    P_hist    = np.zeros((N_MC, n_steps+1, N_DRONES, 8))

    for run, seed in enumerate(seeds):
        err_pos, P_pos_m, _, err8, Pdiag8 = run_une_methode(methode, seed)
        err_hist[run] = err8
        P_hist[run]   = Pdiag8
        for j in range(N_DRONES):
            all_mse[run, j] = np.mean(np.sum(err_pos[IDX_CONV:, j, :]**2, axis=1))
            nci_vals = []
            for k in range(1, n_steps+1):
                e   = err_pos[k, j]
                Pij = P_pos_m[k, j]
                try:
                    d2 = float(e @ inv(Pij) @ e)
                except Exception:
                    continue
                nci_vals.append(d2)
                n_total[j] += 1
                if d2 <= SEUIL_3SIGMA:
                    n_dans_3s[j] += 1
            all_nci[run, j] = np.mean(nci_vals) if nci_vals else np.nan

    couverture   = n_dans_3s / np.maximum(n_total, 1)
    taux_diverg  = np.mean(all_mse > SEUIL_DIVERG, axis=0)   # fraction de runs divergents par drone
    return {
        "mse"         : all_mse,
        "mse_median"  : np.median(all_mse, axis=0),
        "nci"         : all_nci,
        "couverture"  : couverture,
        "taux_diverg" : taux_diverg,
        "err_hist"    : err_hist,
        "P_hist"      : P_hist,
        "methode"     : methode,
        "cout"        : cout_floats(methode),
    }

# =========================================================================
# 8. Figure : 8 variables d'état en colonnes + couloir ±3σ + runs MC
# =========================================================================
labels8 = ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'bx', 'by']
labels_drone = ["Drone 1 (GPS)", "Drone 2 (IMU)", "Drone 3 (sans capteur absolu)"]

def figure_couloirs_methode(r, temps, drone_idx, save_path=None):
    err_hist = r["err_hist"]; P_hist = r["P_hist"]
    n_mc = err_hist.shape[0]; m = r["methode"]
    fig, axs = plt.subplots(4, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{m.nom} — {labels_drone[drone_idx]} — {n_mc} runs Monte-Carlo",
                 fontsize=13, fontweight='bold')
    axs = axs.flatten()
    for i in range(8):
        sigma = np.sqrt(P_hist[0, :, drone_idx, i])         # couloir du 1er run (représentatif)
        for run in range(n_mc):
            axs[i].plot(temps, err_hist[run, :, drone_idx, i],
                        color='green', alpha=0.15, lw=0.6,
                        label='Runs Monte-Carlo' if run == 0 else '_nolegend_')
        axs[i].fill_between(temps, -3*sigma, 3*sigma, color='blue', alpha=0.15,
                            label=r'Couloir $\pm 3\sigma$ prédit')
        axs[i].axhline(0, color='k', lw=0.6)
        axs[i].set_title(f"{labels8[i]} : estimé − vrai", fontsize=10)
        axs[i].grid(True, linestyle=':', alpha=0.7)
    axs[6].set_xlabel("Temps (s)"); axs[7].set_xlabel("Temps (s)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=110)
    return fig

# =========================================================================
# 9. Programme principal
# =========================================================================
if __name__ == "__main__":
    N_MC  = 40
    seeds = list(range(N_MC))
    temps = np.arange(n_steps+1) * dt

    print(f"Étude sur {N_MC} runs Monte-Carlo, {n_steps} pas chacun.")
    print("─" * 60)

    resultats = {}
    for m in METHODES:
        print(f"  Calcul {m.nom} ...", flush=True)
        resultats[m.label] = calcule_metriques(m, N_MC, seeds)

    # ---- Tableau MSE par drone (moyenne, médiane) ----
    print("\n" + "═"*86)
    print(f"MSE position (m²) — fenêtre convergée t >= {T_CONV:.1f}s — {N_MC} runs")
    print("═"*86)
    print(f"{'Méthode':<8}" + "".join(f"{d:>26}" for d in labels_drone))
    print("-"*86)
    for lab in ["M0", "M1", "M2", "M3"]:
        r = resultats[lab]
        moy = r["mse"].mean(axis=0); med = r["mse_median"]
        ligne = f"{lab:<8}"
        for j in range(3):
            ligne += f"   moy={moy[j]:7.2f} med={med[j]:6.2f}"
        print(ligne)
    print("(moy = moyenne inter-runs, sensible aux divergences ; med = médiane, robuste)")

    # ---- Tableau NCI + couverture + taux de divergence ----
    print("\n" + "═"*86)
    print("COHÉRENCE — NCI (cible ≈ 2), couverture 3σ (cible ≈ 99.7%), taux de divergence")
    print("═"*86)
    print(f"{'Méthode':<8}{'Drone':<28}{'NCI':>8}{'Couv.3σ':>10}{'Diverg.':>10}")
    print("-"*86)
    for lab in ["M0", "M1", "M2", "M3"]:
        r = resultats[lab]
        nci = r["nci"].mean(axis=0); couv = r["couverture"]; div = r["taux_diverg"]
        for j in range(3):
            tag = lab if j == 0 else ""
            print(f"{tag:<8}{labels_drone[j]:<28}{nci[j]:>8.2f}{couv[j]*100:>9.1f}%{div[j]*100:>9.1f}%")
        print("-"*86)

    # ---- Figures couloirs : 8 variables x chaque méthode x chaque drone ----
    print("\nGénération des figures de couloirs ±3σ ...")
    for m in METHODES:
        for drone_idx in range(N_DRONES):
            fpath = f"couloirs_{m.label}_drone{drone_idx+1}.png"
            figure_couloirs_methode(resultats[m.label], temps, drone_idx, save_path=fpath)
            plt.close('all')
            print(f"  {fpath}")

    # ---- Boxplot MSE par run (illustre les divergences) ----
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    for j in range(3):
        data = [resultats[lab]["mse"][:, j] for lab in ["M0", "M1", "M2", "M3"]]
        axs[j].boxplot(data, tick_labels=["M0", "M1", "M2", "M3"], showfliers=True)
        axs[j].set_title(labels_drone[j], fontsize=11)
        axs[j].set_ylabel("MSE position (m²)")
        axs[j].grid(True, axis='y', linestyle=':', alpha=0.6)
    fig.suptitle("Distribution des MSE par run — les points isolés = runs divergents",
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("boxplot_mse_par_run.png", dpi=110)
    print("  boxplot_mse_par_run.png")

    print("\nTerminé.")
