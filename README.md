import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import inv, eigvalsh
from scipy.linalg import block_diag
from scipy.optimize import minimize_scalar

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
    if drone_id == 2:
        print("Distribué - Q ax2 =", Q[4,4])
    return Q

def make_P_local():
    P = np.eye(8)
    P[0,0] = P[1,1] = sigma_P_x**2
    P[2,2] = P[3,3] = sigma_P_v**2
    P[4,4] = P[5,5] = sigma_P_a**2
    P[6,6] = P[7,7] = sigma_P_b**2
    return P


class MethodeBloc2x2:
    """M0 — Bloc exact 2x2 (référence)."""
    nom   = "M0 — Bloc 2×2 exact"
    label = "M0"
    color = "black"
    ls    = "-"
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
    n_floats_cov = 1

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
        return {"pos": drone.x[0:2].copy(), "data": drone.P[0:2, 0:2].copy()}

    @staticmethod
    def var_voisin(paquet, Hj):
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

def cout_floats(methode):
    return 3 * (2 + methode.n_floats_cov)

class DroneDistribue:
    def __init__(self, drone_id, x_init, P_init, Q_local):
        self.id = drone_id
        self.x = x_init.copy()
        self.P = P_init.copy()
        self.Q = Q_local.copy()
        self.F = F_loc[drone_id]
        self.B = B_loc[drone_id]

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

            var_j = methode.var_voisin(paquet, Hj)

            if diagnostics is not None and "Ppos_exact" in paquet:
                angle = np.degrees(np.arctan2(Hj[1], Hj[0])) % 360.0
                var_exacte = float(Hj @ paquet["Ppos_exact"] @ Hj.T)
                diagnostics.append((angle, var_j, var_exacte))

            rows_Hi.append(Hi)
            innovs.append(d_mesure - d_pred)
            R_diag.append(R_scalaire + var_j)

        H = np.vstack(rows_Hi)
        innov = np.array(innovs)
        Rv = np.diag(R_diag)

        def cout_CI(omega):
            Pi = self.P / omega
            S = H @ Pi @ H.T + Rv / (1.0 - omega)
            K = Pi @ H.T @ inv(S)
            return np.trace(Pi - K @ H @ Pi)

        sol = minimize_scalar(cout_CI, bounds=(0.01, 0.99), method='bounded')
        omega = sol.x
        Pi = self.P / omega
        S = H @ Pi @ H.T + Rv / (1.0 - omega)
        K = Pi @ H.T @ inv(S)
        self.x = self.x + K @ innov
        self.P = Pi - K @ H @ Pi

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

    err_pos  = np.zeros((n_steps+1, N_DRONES, 2))   
    P_pos_tr = np.zeros((n_steps+1, N_DRONES))      
    P_pos_m  = np.zeros((n_steps+1, N_DRONES, 2, 2))

    # Diagnostic variance/angle : liste de (angle_deg, var_utilisee, var_exacte)
    diag_var = []

    phi_x = phi_y = 0.0

    for k in range(1, n_steps+1):
        step = k - 1
        t    = k * dt

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
            
            for i in (1, 2, 3):
                paquets[i]["Ppos_exact"] = drones[i].P[0:2, 0:2].copy()

            drones[1].update_CI_batch([(z12, paquets[2]), (z13, paquets[3])],
                                       sigma_R_d**2, methode, diagnostics=diag_var)
            drones[2].update_CI_batch([(z12, paquets[1]), (z23, paquets[3])],
                                       sigma_R_d**2, methode, diagnostics=diag_var)
            drones[3].update_CI_batch([(z23, paquets[2]), (z13, paquets[1])],
                                       sigma_R_d**2, methode, diagnostics=diag_var)

        for j, (d, b) in enumerate([(1,0),(2,8),(3,16)]):
            err_pos[k, j]   = drones[d].x[0:2] - X_vrai[b:b+2]
            P_pos_tr[k, j]  = drones[d].P[0,0] + drones[d].P[1,1]
            P_pos_m[k, j]   = drones[d].P[0:2, 0:2]

    return err_pos, P_pos_tr, P_pos_m, np.array(diag_var) if diag_var else np.empty((0,3))

from scipy.stats import chi2
SEUIL_3SIGMA = chi2.ppf(0.997, df=2)   # ~11.6 : couverture cible 99.7%

def calcule_metriques(methode, N_MC=30, seeds=None):

    if seeds is None:
        seeds = range(N_MC)

    all_mse    = np.zeros((N_MC, N_DRONES))
    all_nci    = np.zeros((N_MC, N_DRONES))
    all_trace  = np.zeros((N_MC, n_steps+1, N_DRONES))
    all_rmse_t = np.zeros((N_MC, n_steps+1, N_DRONES))
    nci_min    = np.full(N_DRONES, np.inf)
    n_dans_3s  = np.zeros(N_DRONES)
    n_total    = np.zeros(N_DRONES)
    diag_all   = []

    for run, seed in enumerate(seeds):
        err_pos, P_pos_tr, P_pos_m, diag_var = run_une_methode(methode, seed)
        if diag_var.size:
            diag_all.append(diag_var)

        for j in range(N_DRONES):
            all_mse[run, j] = np.mean(np.sum(err_pos[1:, j, :]**2, axis=1))

            all_rmse_t[run, :, j] = np.sqrt(np.sum(err_pos[:, j, :]**2, axis=1))

            nci_vals = []
            for k in range(1, n_steps+1):
                e   = err_pos[k, j]
                Pij = P_pos_m[k, j]
                try:
                    d2 = float(e @ inv(Pij) @ e)
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

def moy3(metrique, drone):
    """Moyenne sur les 3 drones d'une métrique (N_MC,3) ou (3,)."""
    arr = metrique[drone]
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

labels_drone = ["Drone 1 (GPS)", "Drone 2 (IMU)", "Drone 3 (sans capteur)"]

print("\n\n┌" + "─"*70 + "┐")
print("│  TABLEAU MSE (m²) — erreur quadratique moyenne de position" + " "*12 + "│")
print("└" + "─"*70 + "┘")
print(f"{'Méthode':<22}" + "".join(f"{d:>17}" for d in labels_drone))
print("-"*73)
for label, r in resultats.items():
    mse_par_drone = r["mse"].mean(axis=0)
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
    nci_par_drone = r["nci"].mean(axis=0)
    ligne = f"{r['methode'].label:<6}"
    ligne += "".join(f"{v:>17.3f}" for v in nci_par_drone)
    print(ligne)
print()

fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
fig1.suptitle("Erreur de variance selon l'orientation du lien",
              fontsize=12, fontweight='bold')

ecarts_tous = np.concatenate([resultats[m.label]["diag_var"][:, 1]
                              - resultats[m.label]["diag_var"][:, 2]
                              for m in METHODES])
y_lim = np.percentile(np.abs(ecarts_tous), 98)

bins = np.arange(0, 361, 15)
mids = (bins[:-1] + bins[1:]) / 2

for ax, m in zip(axes1, METHODES):
    dv    = resultats[m.label]["diag_var"]
    angle = dv[:, 0]
    ecart = dv[:, 1] - dv[:, 2]

    med, q1, q3 = [], [], []
    for k in range(len(bins) - 1):
        sel = (angle >= bins[k]) & (angle < bins[k+1])
        if np.any(sel):
            med.append(np.median(ecart[sel]))
            q1.append(np.percentile(ecart[sel], 25))
            q3.append(np.percentile(ecart[sel], 75))
        else:
            med.append(np.nan); q1.append(np.nan); q3.append(np.nan)
    med, q1, q3 = np.array(med), np.array(q1), np.array(q3)

    ax.fill_between(mids, q1, q3, color=m.color, alpha=0.20,
                    label='Écart inter-quartiles')
    ax.plot(mids, med, color=m.color, lw=2.5, marker='o', ms=3,
            zorder=5, label='Médiane')

    ax.axhline(0, color='black', lw=1)
    ax.set_xlim(0, 360); ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_ylim(-y_lim, y_lim)
    ax.set_title(m.label, fontsize=12, color=m.color, fontweight='bold')
    ax.set_xlabel("Angle du lien (°)")
    ax.grid(True, linestyle=':', alpha=0.4)

axes1[0].set_ylabel("Variance utilisée − exacte")
axes1[0].legend(fontsize=7, loc='upper left')
fig1.tight_layout()

def normalise(vals):
    arr = np.array(vals, dtype=float)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)

crit_mse  = normalise([resultats[m.label]["mse"].mean() for m in METHODES])
crit_nci  = normalise([abs(resultats[m.label]["nci"].mean() - 2) for m in METHODES])
crit_geo  = normalise([resultats[m.label]["var_err_var"] for m in METHODES])
crit_cout = normalise([resultats[m.label]["cout"] for m in METHODES])

scores = (crit_mse + crit_nci + crit_geo + crit_cout) / 4.0

ordre = np.argsort(scores)
methodes_triees = [METHODES[i] for i in ordre]
scores_tries    = scores[ordre]

fig2, ax2 = plt.subplots(figsize=(9, 4.5))
y_pos = np.arange(len(methodes_triees))
barres = ax2.barh(y_pos, scores_tries,
                  color=[m.color for m in methodes_triees], alpha=0.85)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([m.nom for m in methodes_triees], fontsize=10)
ax2.invert_yaxis()
ax2.set_xlabel("Score global  (0 = meilleur, 1 = pire)", fontsize=11)
ax2.set_title("Classement global des méthodes\n"
              "moyenne de 4 critères : précision, cohérence, géométrie, coût radio",
              fontsize=12, fontweight='bold')
ax2.grid(True, axis='x', linestyle=':', alpha=0.5)

for bar, sc in zip(barres, scores_tries):
    ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f"{sc:.2f}", va='center', fontsize=9, fontweight='bold')
ax2.set_xlim(0, 1.1)
fig2.tight_layout()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.linalg import block_diag

# --------------------------------------------------------------------------
# 1. Paramètres de simulation
# --------------------------------------------------------------------------
t_max      = 16.0
dt         = 0.1
dt_capteur = 0.5
dt_imu     = 0.1
n_drone    = 3
n_var      = 8
N          = n_drone * n_var
n_steps    = int(round(t_max / dt))

ratio_imu = int(round(dt_imu / dt))
ratio_gps = int(round(dt_capteur / dt))

# --------------------------------------------------------------------------
# 2. Écarts-types
# --------------------------------------------------------------------------
sigma_w_accel = 5e-2
sigma_w_autre = 1e-6

sigma_gps = 0.5
sigma_acc = 0.01
sigma_d   = 0.5

sigma_R_gps = 0.5
sigma_R_acc = 0.1
sigma_R_d   = 0.5

sigma_P_x, sigma_P_v, sigma_P_a, sigma_P_b = 2.0, 0.5, 0.5, 1.0

# --------------------------------------------------------------------------
# 3. Matrices du modèle
# --------------------------------------------------------------------------
def Bmat(cols):
    M = np.zeros((8, 6))
    M[4, cols[0]] = 1.0
    M[5, cols[1]] = 1.0
    return M

def Fmat(accel_constante):
    a = 1.0 if accel_constante else 0.0
    return np.array([
        [1, 0, dt, 0, 0.5*dt*dt, 0,        0, 0],
        [0, 1, 0,  dt, 0,        0.5*dt*dt, 0, 0],
        [0, 0, 1,  0,  dt,       0,         0, 0],
        [0, 0, 0,  1,  0,        dt,        0, 0],
        [0, 0, 0,  0,  a,        0,         0, 0],
        [0, 0, 0,  0,  0,        a,         0, 0],
        [0, 0, 0,  0,  0,        0,         1, 0],
        [0, 0, 0,  0,  0,        0,         0, 1]], dtype=float)

F_vrai = block_diag(Fmat(False), Fmat(False), Fmat(False))
B_vrai = np.concatenate((Bmat([0, 1]), Bmat([2, 3]), Bmat([4, 5])), axis=0)
F_kalman = block_diag(Fmat(False), Fmat(True), Fmat(False))
B_kalman = np.concatenate((Bmat([0, 1]), np.zeros((8, 6)), Bmat([4, 5])), axis=0)

I_N = np.eye(N)

w_sigma = np.full(N, sigma_w_autre)
for b in (0, 8, 16):
    w_sigma[b+4] = w_sigma[b+5] = sigma_w_accel

X_vrai_init = np.concatenate(([0,  10, 1, 0, 0, 0,  0,    0],
                               [10,  0, 1, 0, 0, 0,  0.5, -0.2],
                               [0, -10, 1, 0, 0, 0,  0,    0])).astype(float)

# --------------------------------------------------------------------------
# 4. Helper EKF — forme de Joseph
# --------------------------------------------------------------------------
def maj_kalman(X, P, H, innov, R):
    S = H @ P @ H.T + R
    K = P @ H.T @ inv(S)
    X = X + K @ innov
    A = I_N - K @ H
    P = A @ P @ A.T + K @ R @ K.T
    return X, P

# --------------------------------------------------------------------------
# 5. Figures utilitaires
# --------------------------------------------------------------------------
labels = ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'bx', 'by']

def figure_drone(nom_scenario, d, base, tv, tk, P_hist, temps, mes_gps, mes_imu, mes_gpsv, titre_suffix=""):
    fig, axs = plt.subplots(4, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{nom_scenario} - Drone {d}",
                 fontsize=13, fontweight='bold')
    axs = axs.flatten()
    for i in range(8):
        idx   = base + i
        sigma = np.sqrt(P_hist[:, idx, idx])
        err   = tk[:, idx] - tv[:, idx]
        axs[i].fill_between(temps, -3*sigma, 3*sigma, color='blue', alpha=0.2,
                            label=r'Couloir $\pm 3\sigma$')
        axs[i].plot(temps, err, color='green', label='Erreur EKF')
        axs[i].axhline(0, color='k', lw=0.6)
        axs[i].set_title(f"{labels[i]} : estimé − vrai", fontsize=10)
        axs[i].grid(True, linestyle=':', alpha=0.7)
    if d == 1 and len(mes_gps):
        axs[0].scatter(mes_gps[:, 0], mes_gps[:, 1] - mes_gpsv[:, 0],
                       color='red', marker='x', s=20, label='Mesure GPS')
        axs[1].scatter(mes_gps[:, 0], mes_gps[:, 2] - mes_gpsv[:, 1],
                       color='red', marker='x', s=20, label='Mesure GPS')
    if d == 2 and len(mes_imu):
        axs[4].scatter(mes_imu[:, 0], mes_imu[:, 1] - mes_imu[:, 3],
                       color='red', marker='x', s=20, label='Mesure IMU')
        axs[5].scatter(mes_imu[:, 0], mes_imu[:, 2] - mes_imu[:, 4],
                       color='red', marker='x', s=20, label='Mesure IMU')
    axs[6].set_xlabel("Temps (s)"); axs[7].set_xlabel("Temps (s)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

def figure_comparaison_biais(tv, tk_avec, tk_sans, temps, label_avec, label_sans):
    fig, axs = plt.subplots(2, 2, figsize=(13, 7))
    fig.suptitle("Impact de l'estimation du biais — Drone 2  (bx=0.5, by=-0.2)",
                 fontsize=13, fontweight='bold')
    axs[0,0].plot(temps, tv[:,8],      'k',   lw=1.5, label='Vérité')
    axs[0,0].plot(temps, tk_avec[:,8], 'g',   lw=1.5, label=label_avec)
    axs[0,0].plot(temps, tk_sans[:,8], 'r--', lw=1.5, label=label_sans)
    axs[0,0].set_title("Position x — Drone 2"); axs[0,0].set_ylabel("x (m)")
    axs[0,0].legend(); axs[0,0].grid(True, linestyle=':', alpha=0.7)

    axs[0,1].plot(temps, tv[:,9],      'k',   lw=1.5)
    axs[0,1].plot(temps, tk_avec[:,9], 'g',   lw=1.5)
    axs[0,1].plot(temps, tk_sans[:,9], 'r--', lw=1.5)
    axs[0,1].set_title("Position y — Drone 2"); axs[0,1].set_ylabel("y (m)")
    axs[0,1].grid(True, linestyle=':', alpha=0.7)

    axs[1,0].axhline(0.5, color='k', lw=1.5, label='Vrai biais bx = 0.5')
    axs[1,0].plot(temps, tk_avec[:,14], 'g',   lw=1.5, label=label_avec)
    axs[1,0].plot(temps, tk_sans[:,14], 'r--', lw=1.5, label=label_sans)
    axs[1,0].set_title("Estimation du biais bx — Drone 2"); axs[1,0].set_ylabel("bx")
    axs[1,0].set_xlabel("Temps (s)")
    axs[1,0].legend(); axs[1,0].grid(True, linestyle=':', alpha=0.7)

    err_avec = np.sqrt((tv[:,8]-tk_avec[:,8])**2 + (tv[:,9]-tk_avec[:,9])**2)
    err_sans  = np.sqrt((tv[:,8]-tk_sans[:,8])**2 + (tv[:,9]-tk_sans[:,9])**2)
    axs[1,1].plot(temps, err_avec, 'g',   lw=1.5, label=label_avec)
    axs[1,1].plot(temps, err_sans,  'r--', lw=1.5, label=label_sans)
    axs[1,1].set_title("Erreur de position euclidienne — Drone 2")
    axs[1,1].set_ylabel("||erreur|| (m)"); axs[1,1].set_xlabel("Temps (s)")
    axs[1,1].legend(); axs[1,1].grid(True, linestyle=':', alpha=0.7)

    fig.tight_layout()
    return fig

def figure_trajectoires(tv, scenarios, temps, mes_gps):
    fig = plt.figure(figsize=(10, 8))
    i1 = n_steps // 3,
    for d, base, mk in [(1, 0, '^'), (2, 8, 'o'), (3, 16, 's')]:
        plt.plot(tv[:, base], tv[:, base+1], color='black', lw=2,
                label=f'Drone {d} — vérité')
    for tk, label, color, ls in scenarios :
        for d, base, mk in [(1,0, '^'), (2, 8, 'o'), (3, 16, 's')]:
            if d == 1 :
                lbl = label
            else :
                lbl = '_nolegend'
            plt.plot(tk[:, base], tk[:, base+1], color=color, linestyle=ls, lw=1.5,
                    label=lbl)

    plt.xlabel("X"); plt.ylabel("Y"); plt.title("Trajectoires des 3 drones")
    plt.legend(fontsize=8); plt.grid(True)
    x_vrai_all = tv[:, [0, 8, 16]]
    y_vrai_all = tv[:, [1, 9, 17]]
    marge = 10
    plt.xlim(x_vrai_all.min() - marge, x_vrai_all.max() + marge)
    plt.ylim(y_vrai_all.min() - marge, y_vrai_all.max() + marge)
    return fig

def run_ekf(nom_scenario="", compenser_biais=True, seed=0,
            use_gps=True, use_imu=True, use_distances=True,
            show_corridors=False):
    np.random.seed(seed)

    X_vrai = X_vrai_init.copy()

    erreur_init = np.zeros(N)
    for b in (0, 8, 16):
        erreur_init[b:b+2] = np.random.normal(0, sigma_P_x, size=2)
        erreur_init[b+2:b+4] = np.random.normal(0, sigma_P_v, size=2)
        erreur_init[b+4:b+6] = np.random.normal(0, sigma_P_a, size=2)
        erreur_init[b+6:b+8] = np.random.normal(0, sigma_P_b, size=2)
    erreur_init[6:8] = [0,0]
    erreur_init[22:24] = [0,0]
    print(erreur_init)
    X_est = X_vrai + erreur_init

    P_est = np.eye(N)
    for b in (0, 8, 16):
        P_est[b,   b]   = P_est[b+1, b+1] = sigma_P_x**2
        P_est[b+2, b+2] = P_est[b+3, b+3] = sigma_P_v**2
        P_est[b+4, b+4] = P_est[b+5, b+5] = sigma_P_a**2
        P_est[b+6, b+6] = P_est[b+7, b+7] = sigma_P_b**2

    P_est[10,10] = P_est[11,11]  = sigma_P_v**2 + 2
    Q = np.eye(N) * 1e-3
    for b in (0, 8, 16):
        Q[b+4, b+4] = Q[b+5, b+5] = 0.5**2
        Q[b+6, b+6] = Q[b+7, b+7] = 1e-9**2
    Q[10, 10] = Q[11, 11] = 0.1**2
    Q[12, 12] = Q[13, 13] = 1**2

    print("Q ax2", Q[12,12] if N==24 else Q[4,4])

    if not compenser_biais:
        X_est[6:8]   = 0.0
        X_est[14:16] = 0.0
        X_est[22:24] = 0.0
        for i in (6, 7, 14, 15, 22, 23):
            P_est[i, i] = 1e-8
            Q[i, i]     = 1e-8

    traj_vrai   = np.zeros((n_steps + 1, N))
    traj_kalman = np.zeros((n_steps + 1, N))
    P_hist      = np.zeros((n_steps + 1, N, N))
    temps       = np.zeros(n_steps + 1)

    traj_vrai[0]   = X_vrai
    traj_kalman[0] = X_est
    P_hist[0]      = P_est

    mes_gps  = []
    mes_imu  = []
    mes_gpsv = []

    phi_x = phi_y = 0.0

    for k in range(1, n_steps + 1):
        step = k - 1
        t    = k * dt

        phi_x += 5 * dt
        phi_y += 1 * dt
        u_vrai = np.array([np.cos(phi_x), np.sin(phi_y),
                            np.cos(phi_x), np.sin(phi_y),
                            np.cos(phi_x), np.sin(phi_y)])

        # Commande
        if step < n_steps / 3:
            u_vrai = np.array([1., 0., 1., 0., 1., 0.])
        elif step < 2 * n_steps / 3:
            phi_x += 5 * dt
            phi_y += 1 * dt
            u_vrai = np.array([np.cos(phi_x), np.sin(phi_y),
                               np.cos(phi_x), np.sin(phi_y),
                               np.cos(phi_x), np.sin(phi_y)])
        else:
            u_vrai = np.array([1., 0., 1., 0., 1., 0.])

        err_cmd = np.random.normal(0, 0.1, size=6)
        err_cmd[0:2] = 0.0
        u_kalman = u_vrai + err_cmd

        # Propagation de la vérité
        X_vrai = F_vrai @ X_vrai + B_vrai @ u_vrai + np.random.normal(0, 1, N) * w_sigma

        # Prédiction du filtre
        Xc = F_kalman @ X_est + B_kalman @ u_kalman
        Pc = F_kalman @ P_est @ F_kalman.T + Q

        # ---- Correction IMU (drone 2) ------------------------------------
        if use_imu and step % ratio_imu == 0:
            mes = np.array([X_vrai[12] + X_vrai[14] + np.random.normal(0, sigma_acc),
                            X_vrai[13] + X_vrai[15] + np.random.normal(0, sigma_acc)])
            H = np.zeros((2, N))
            H[0, 12] = H[0, 14] = 1.0
            H[1, 13] = H[1, 15] = 1.0
            innov = mes - H @ Xc
            Xc, Pc = maj_kalman(Xc, Pc, H, innov, np.diag([sigma_R_acc**2]*2))
            mes_imu.append((t, mes[0], mes[1], X_vrai[12], X_vrai[13]))

        # ---- Correction GPS + distances (construction dynamique) ---------
        if step % ratio_gps == 0:

            d12 = np.hypot(Xc[0]-Xc[8],  Xc[1]-Xc[9])
            d23 = np.hypot(Xc[8]-Xc[16], Xc[9]-Xc[17])
            d13 = np.hypot(Xc[0]-Xc[16], Xc[1]-Xc[17])

            d12v = np.hypot(X_vrai[0]-X_vrai[8],  X_vrai[1]-X_vrai[9])
            d23v = np.hypot(X_vrai[8]-X_vrai[16], X_vrai[9]-X_vrai[17])
            d13v = np.hypot(X_vrai[0]-X_vrai[16], X_vrai[1]-X_vrai[17])

            meas_vals = []
            h_vals    = []
            H_rows    = []
            R_diag    = []

            if use_gps:
                meas_vals += [X_vrai[0] + np.random.normal(0, sigma_gps),
                              X_vrai[1] + np.random.normal(0, sigma_gps)]
                h_vals    += [Xc[0], Xc[1]]
                H_gps = np.zeros((2, N))
                H_gps[0, 0] = 1.0
                H_gps[1, 1] = 1.0
                H_rows.append(H_gps)
                R_diag += [sigma_R_gps**2, sigma_R_gps**2]
                mes_gps.append((t, meas_vals[0], meas_vals[1]))
                mes_gpsv.append((X_vrai[0], X_vrai[1]))

            if use_distances:
                meas_vals += [d12v + np.random.normal(0, sigma_d),
                              d23v + np.random.normal(0, sigma_d),
                              d13v + np.random.normal(0, sigma_d)]
                h_vals += [d12, d23, d13]
                H_dist = np.zeros((3, N))
                H_dist[0, 0] =  (Xc[0]-Xc[8])  / d12;  H_dist[0, 1] =  (Xc[1]-Xc[9])  / d12
                H_dist[0, 8] = -H_dist[0, 0];           H_dist[0, 9] = -H_dist[0, 1]
                H_dist[1, 8] =  (Xc[8]-Xc[16]) / d23;  H_dist[1, 9] =  (Xc[9]-Xc[17]) / d23
                H_dist[1,16] = -H_dist[1, 8];           H_dist[1,17] = -H_dist[1, 9]
                H_dist[2, 0] =  (Xc[0]-Xc[16]) / d13;  H_dist[2, 1] =  (Xc[1]-Xc[17]) / d13
                H_dist[2,16] = -H_dist[2, 0];           H_dist[2,17] = -H_dist[2, 1]
                H_rows.append(H_dist)
                R_diag += [sigma_R_d**2, sigma_R_d**2, sigma_R_d**2]

            if H_rows:
                mes    = np.array(meas_vals)
                h_pred = np.array(h_vals)
                H      = np.vstack(H_rows)
                R      = np.diag(R_diag)
                innov  = mes - h_pred
                Xc, Pc = maj_kalman(Xc, Pc, H, innov, R)

        #Sauvegarde des données
        X_est, P_est = Xc, Pc
        traj_vrai[k]   = X_vrai
        traj_kalman[k] = X_est
        P_hist[k]      = P_est
        temps[k]       = t

    mes_gps  = np.array(mes_gps)  if mes_gps  else np.empty((0, 3))
    mes_imu  = np.array(mes_imu)  if mes_imu  else np.empty((0, 5))
    mes_gpsv = np.array(mes_gpsv) if mes_gpsv else np.empty((0, 2))

    if show_corridors:
        titre = f"({'biais estimé' if compenser_biais else 'biais non estimé'}, " \
                f"GPS={'on' if use_gps else 'off'}, " \
                f"IMU={'on' if use_imu else 'off'}, " \
                f"dist={'on' if use_distances else 'off'})"
        for d, base in [(1, 0), (2, 8), (3, 16)]:
            figure_drone(nom_scenario, d, base, traj_vrai, traj_kalman, P_hist, temps,
                         mes_gps, mes_imu, mes_gpsv, titre_suffix=titre)

    return traj_vrai, traj_kalman, P_hist, temps, mes_gps, mes_imu, mes_gpsv

def run_monte_carlo(n_mc=50, base_seed=1000, nom_scenario="", **kwargs):

    kwargs.pop('show_corridors', None)
    kwargs.pop('seed', None)
    kwargs.pop('nom_scenario', None)

    tk_all  = np.zeros((n_mc, n_steps + 1, N))
    err_all = np.zeros((n_mc, n_steps + 1, N))
    P_ref   = None
    tv_ref  = None
    temps   = None

    for i in range(n_mc):
        tv, tk, Ph, temps, *_ = run_ekf(
            seed=base_seed + i, show_corridors=False, **kwargs)
        tk_all[i]  = tk
        err_all[i] = tk - tv
        if P_ref is None:
            P_ref  = Ph
            tv_ref = tv

    return tk_all, err_all, P_ref, tv_ref, temps


def figure_mc_consistance(err_all, P_ref, temps, base, nom="", drone=2):

    n_mc = err_all.shape[0]
    fig, axs = plt.subplots(4, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{nom} — Drone {drone} — {n_mc} runs Monte-Carlo",
                 fontsize=13, fontweight='bold')
    axs = axs.flatten()
    for i in range(8):
        idx   = base + i
        sigma = np.sqrt(P_ref[:, idx, idx])
        rmse  = np.sqrt(np.mean(err_all[:, :, idx]**2, axis=0))
        for r in range(n_mc):
            axs[i].plot(temps, err_all[r, :, idx],
                        color='green', alpha=0.15, lw=0.6, label='Runs Monte-Carlo' if r==0 else '_nolegend_')
        axs[i].fill_between(temps, -3*sigma, 3*sigma, color='blue', alpha=0.15,
                            label=r'Couloir $\pm 3\sigma$ prédit')
        # axs[i].plot(temps,  rmse, 'r-', lw=1.5, label='RMSE empirique')
        # axs[i].plot(temps, -rmse, 'r-', lw=1.5)
        axs[i].axhline(0, color='k', lw=0.6)
        axs[i].set_title(f"{labels[i]} : estimé − vrai", fontsize=10)
        axs[i].grid(True, linestyle=':', alpha=0.7)
    axs[6].set_xlabel("Temps (s)"); axs[7].set_xlabel("Temps (s)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def figure_mc_rmse_position(scenarios_mc, temps, drone=2):

    base = {1: 0, 2: 8, 3: 16}[drone]
    fig, ax = plt.subplots(figsize=(10, 5))
    for err_all, label, color in scenarios_mc:
        # erreur de position euclidienne, par run, à chaque instant
        err_pos = np.sqrt(err_all[:, :, base]**2 + err_all[:, :, base+1]**2)
        rmse    = np.sqrt(np.mean(err_pos**2, axis=0))   # RMSE inter-runs
        lo      = err_pos.min(axis=0)
        hi      = err_pos.max(axis=0)
        ax.fill_between(temps, lo, hi, color=color, alpha=0.12)
        ax.plot(temps, rmse, color=color, lw=1.8, label=label)
    ax.set_title(f"RMSE de position — Drone {drone} (bande = min/max des runs)",
                 fontsize=12)
    ax.set_xlabel("Temps (s)"); ax.set_ylabel("RMSE (m)")
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.7)
    fig.tight_layout()
    return fig

MODE_MONTE_CARLO = False 
N_MC             = 50 
BASE_SEED_MC     = 1000

DRONES_A_TRACER  = [(1, 0), (2, 8), (3, 16)]

CONFIGS = [
    ("Scénario D", dict(compenser_biais=True,  use_gps=True, use_imu=True, use_distances=True)),
    ("Scénario C", dict(compenser_biais=False, use_gps=True, use_imu=True, use_distances=True)),
    ("Scénario A", dict(compenser_biais=False, use_gps=True, use_imu=True, use_distances=False)),
    ("Scénario B", dict(compenser_biais=True,  use_gps=True, use_imu=True, use_distances=False)),
]

if MODE_MONTE_CARLO:

    print(f"Mode Monte-Carlo : {N_MC} runs par scénario...\n")

    err_mc   = {}
    Pref_mc  = {}
    tvref_mc = {}
    temps    = None

    for nom, cfg in CONFIGS:
        print(f"  {nom} ...")
        _, err_all, P_ref, tv_ref, temps = run_monte_carlo(
            n_mc=N_MC, base_seed=BASE_SEED_MC, **cfg)
        err_mc[nom]   = err_all
        Pref_mc[nom]  = P_ref
        tvref_mc[nom] = tv_ref

    print("\n=== MSE position drone 2 (moyenne sur les runs) ===")
    for nom, cfg in CONFIGS:
        mse_runs = np.mean(err_mc[nom][:, :, 8:10]**2)
        print(f"  {nom:<12} : {mse_runs:.4f}")

    for nom, cfg in CONFIGS:
        for drone, base in DRONES_A_TRACER:
            figure_mc_consistance(err_mc[nom], Pref_mc[nom], temps,
                                  base=base, nom=nom, drone=drone)

    figure_mc_rmse_position([
        (err_mc["Scénario D"], "D — Complet, biais estimé",            'green'),
        (err_mc["Scénario C"], "C — Complet, biais non estimé",         'orange'),
        (err_mc["Scénario A"], "A — Sans distances, biais non estimé",  'red'),
        (err_mc["Scénario B"], "B — Sans distances, biais estimé",      'blue'),
    ], temps, drone=2)

    plt.show()

else:

    # --- Scénario D : configuration complète, avec biais estimé (référence) ---
    print("Scénario D : tous capteurs, biais estimé...")
    tv, tk_D, Ph_D, temps, gps_D, imu_D, gpsv_D = run_ekf(
        nom_scenario="Scénario D",
        compenser_biais=True, seed=0,
        use_gps=True, use_imu=True, use_distances=True,
        show_corridors=True)

    # --- Scénario C : tous capteurs, sans estimer le biais ---
    print("Scénario C : tous capteurs, biais NON estimé...")
    _, tk_C, Ph_C, _, gps_C, imu_C, gpsv_C = run_ekf(
        nom_scenario="Scénario C",
        compenser_biais=False, seed=0,
        use_gps=True, use_imu=True, use_distances=True,
        show_corridors=True)

    # --- Scénario A : sans distances, sans estimer le biais (dérive parabolique) ---
    print("Scénario A : sans distances, biais NON estimé...")
    _, tk_A, Ph_A, _, gps_A, imu_A, gpsv_A = run_ekf(
        nom_scenario="Scénario A",
        compenser_biais=False, seed=0,
        use_gps=True, use_imu=True, use_distances=False,
        show_corridors=True)

    # --- Scénario B : sans distances, avec biais estimé ---
    print("Scénario B : sans distances, biais estimé...")
    _, tk_B, Ph_B, _, gps_B, imu_B, gpsv_B = run_ekf(
        nom_scenario="Scénario B",
        compenser_biais=True, seed=0,
        use_gps=True, use_imu=True, use_distances=False,
        show_corridors=True)

        
    mse = lambda a, b: np.square(a - b).mean()
    print("\n=== MSE position drone 2 ===")
    for label, tk in [("D - Complet + biais estimé",           tk_D),
                      ("C - Complet, biais non estimé",        tk_C),
                      ("A - Sans distances, biais non estimé", tk_A),
                      ("B - Sans distances, biais estimé",     tk_B)]:
        print(f"  {label:<40} : {mse(tv[:,8:10], tk[:,8:10]):.4f}")

    # Figure — Comparaison C vs D : distances compensent le biais
    f2 = figure_comparaison_biais(tv, tk_C, tk_D, temps,
                                   label_avec="Biais estimé (D)",
                                   label_sans="Biais non estimé (C)")
    f2.suptitle("Scénario C vs D", fontsize=13, fontweight='bold')

    # Figure — Comparaison A vs B : sans distances, la dérive parabolique apparaît
    f3 = figure_comparaison_biais(tv, tk_A, tk_B, temps,
                                   label_avec="Biais estimé (B)",
                                   label_sans="Biais non estimé (A)")
    f3.suptitle("Scénario A vs B", fontsize=13, fontweight='bold')

    # Figure — 4 scénarios, erreur de position drone 2
    fig4, ax = plt.subplots(figsize=(10, 5))
    for tk, label, color, ls in [
        (tk_D, "D — Complet, biais estimé",           'green',  '-'),
        (tk_C, "C — Complet, biais non estimé",        'orange', '--'),
        (tk_A, "A — Sans distances, biais non estimé", 'red',    '-.'),
        (tk_B, "B — Sans distances, biais estimé",     'blue',   ':'),
    ]:
        err = np.sqrt((tv[:,8]-tk[:,8])**2 + (tv[:,9]-tk[:,9])**2)
        ax.plot(temps, err, color=color, linestyle=ls, lw=1.8, label=label)
    ax.set_title("Comparaison 4 scénarios", fontsize=12)
    ax.set_xlabel("Temps (s)"); ax.set_ylabel("erreur (m)")
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.7)
    fig4.tight_layout()

    # Figure — Trajectoires scénario de référence
    f5 = figure_trajectoires(tv, [
        (tk_D, 'Scénario D', 'green', '-'),
        (tk_C, 'Scénario C', 'orange', '--'),
        (tk_A, 'Scénario A', 'red', '-.'),
        (tk_B, 'Scénario B', 'blue', ':'),
    ], temps, gps_D)

    plt.show()
