"""
============================================================================
 EKF — Localisation coopérative de 3 drones
 Capteurs activables / désactivables pour la présentation des résultats
 + Mode Monte-Carlo activable (MODE_MONTE_CARLO en section 7)
============================================================================
 Paramètres de run_ekf() :
   compenser_biais : bool  — le filtre estime-t-il le biais ?
   use_gps         : bool  — GPS sur le drone 1 actif ?
   use_imu         : bool  — IMU (accéléromètre) sur le drone 2 actif ?
   use_distances   : bool  — capteurs de distance inter-drones actifs ?
   show_corridors  : bool  — afficher les figures ±3σ pour les 3 drones ?
   seed            : int   — graine aléatoire (même seed = même trajectoire vraie)
============================================================================
"""
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

F_vrai   = block_diag(Fmat(False), Fmat(False), Fmat(False))
B_vrai   = np.concatenate((Bmat([0, 1]), Bmat([2, 3]), Bmat([4, 5])), axis=0)
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
    # if len(mes_gps):
    #     plt.scatter(mes_gps[:, 1], mes_gps[:, 2], color='red', marker='x',
    #                 s=20, label='Mesures GPS (Drone 1)', zorder=5)

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

# --------------------------------------------------------------------------
# 6. Fonction principale
# --------------------------------------------------------------------------
def run_ekf(nom_scenario="", compenser_biais=True, seed=0,
            use_gps=True, use_imu=True, use_distances=True,
            show_corridors=False):
    np.random.seed(seed)

    # ---- Initialisation --------------------------------------------------
    X_vrai = X_vrai_init.copy()

    erreur_init = np.zeros(N)
    for b in (0, 8, 16):
        erreur_init[b:b+2] = np.random.normal(0, 2.0, size=2)
    X_est = X_vrai + erreur_init

    P_est = np.eye(N)
    for b in (0, 8, 16):
        P_est[b,   b]   = P_est[b+1, b+1] = sigma_P_x**2
        P_est[b+2, b+2] = P_est[b+3, b+3] = sigma_P_v**2
        P_est[b+4, b+4] = P_est[b+5, b+5] = sigma_P_a**2
        P_est[b+6, b+6] = P_est[b+7, b+7] = sigma_P_b**2

    Q = np.eye(N) * 1e-3
    for b in (0, 8, 16):
        Q[b+4, b+4] = Q[b+5, b+5] = 0.5**2
        Q[b+6, b+6] = Q[b+7, b+7] = 1e-5**2
    Q[10, 10] = Q[11, 11] = 0.1**2
    Q[12, 12] = Q[13, 13] = 1e-2**2

    # ---- Gel du biais si demandé ----------------------------------------
    if not compenser_biais:
        X_est[6:8]   = 0.0
        X_est[14:16] = 0.0
        X_est[22:24] = 0.0
        for i in (6, 7, 14, 15, 22, 23):
            P_est[i, i] = 1e-8
            Q[i, i]     = 1e-8

    # ---- Historique ------------------------------------------------------
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

    # ---- Boucle principale -----------------------------------------------
    phi_x = phi_y = 0.0

    for k in range(1, n_steps + 1):
        step = k - 1
        t    = k * dt

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

        # Enregistrement
        X_est, P_est = Xc, Pc
        traj_vrai[k]   = X_vrai
        traj_kalman[k] = X_est
        P_hist[k]      = P_est
        temps[k]       = t

    mes_gps  = np.array(mes_gps)  if mes_gps  else np.empty((0, 3))
    mes_imu  = np.array(mes_imu)  if mes_imu  else np.empty((0, 5))
    mes_gpsv = np.array(mes_gpsv) if mes_gpsv else np.empty((0, 2))

    # ---- Figures couloirs ±3σ si demandé --------------------------------
    if show_corridors:
        titre = f"({'biais estimé' if compenser_biais else 'biais non estimé'}, " \
                f"GPS={'on' if use_gps else 'off'}, " \
                f"IMU={'on' if use_imu else 'off'}, " \
                f"dist={'on' if use_distances else 'off'})"
        for d, base in [(1, 0), (2, 8), (3, 16)]:
            figure_drone(nom_scenario, d, base, traj_vrai, traj_kalman, P_hist, temps,
                         mes_gps, mes_imu, mes_gpsv, titre_suffix=titre)

    return traj_vrai, traj_kalman, P_hist, temps, mes_gps, mes_imu, mes_gpsv

# --------------------------------------------------------------------------
# 9. Couche Monte-Carlo
# --------------------------------------------------------------------------
def run_monte_carlo(n_mc=50, base_seed=1000, nom_scenario="", **kwargs):
    """Exécute n_mc runs EKF indépendants (seeds différents).

    kwargs = compenser_biais, use_gps, use_imu, use_distances...
    show_corridors et seed sont forcés/ignorés ici : on trace nous-mêmes.

    Retourne :
      tk_all  (n_mc, n_steps+1, N) : trajectoires estimées
      err_all (n_mc, n_steps+1, N) : erreurs (estimé - vrai) par run
      P_ref   (n_steps+1, N, N)    : un P représentatif (σ prédit par le filtre)
      tv_ref  (n_steps+1, N)       : une trajectoire vraie de référence (1 run)
      temps   (n_steps+1,)
    """
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
            # P dépend de la géométrie (H) plus que des valeurs de bruit :
            # un run sert de référence représentative pour le couloir ±3σ.
            P_ref  = Ph
            tv_ref = tv

    return tk_all, err_all, P_ref, tv_ref, temps


def figure_mc_consistance(err_all, P_ref, temps, base, nom="", drone=2):
    """Diagnostic de cohérence : spaghetti des erreurs de tous les runs,
    RMSE empirique (rouge) et couloir ±3σ prédit par le filtre (bleu).

    Lecture : si le filtre est cohérent, la RMSE empirique reste dans le
    couloir ±3σ et n'en sort pas. Si elle déborde, le filtre est trop
    confiant (sous-estime son incertitude réelle)."""
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
                        color='green', alpha=0.08, lw=0.6)
        axs[i].fill_between(temps, -3*sigma, 3*sigma, color='blue', alpha=0.15,
                            label=r'Couloir $\pm 3\sigma$ prédit')
        axs[i].plot(temps,  rmse, 'r-', lw=1.5, label='RMSE empirique')
        axs[i].plot(temps, -rmse, 'r-', lw=1.5)
        axs[i].axhline(0, color='k', lw=0.6)
        axs[i].set_title(f"{labels[i]} : estimé − vrai", fontsize=10)
        axs[i].grid(True, linestyle=':', alpha=0.7)
    axs[6].set_xlabel("Temps (s)"); axs[7].set_xlabel("Temps (s)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def figure_mc_rmse_position(scenarios_mc, temps, drone=2):
    """Synthèse : RMSE de position euclidienne (drone choisi) par scénario,
    avec bande de dispersion inter-runs (min/max). Idéale en diapo bilan."""
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

# --------------------------------------------------------------------------
# 7. Scénarios de présentation
# --------------------------------------------------------------------------

MODE_MONTE_CARLO = True    # True = exécution Monte-Carlo ; False = single-run d'origine
N_MC             = 50      # nombre de runs Monte-Carlo par scénario
BASE_SEED_MC     = 1000    # seed de départ (incrémenté à chaque run)

# Configuration des 4 scénarios (réutilisée par les deux modes)
CONFIGS = [
    ("Scénario A", dict(compenser_biais=True,  use_gps=True, use_imu=True, use_distances=True)),
    ("Scénario B", dict(compenser_biais=False, use_gps=True, use_imu=True, use_distances=True)),
    ("Scénario C", dict(compenser_biais=False, use_gps=True, use_imu=True, use_distances=False)),
    ("Scénario D", dict(compenser_biais=True,  use_gps=True, use_imu=True, use_distances=False)),
]

if MODE_MONTE_CARLO:
    # ======================================================================
    #  MODE MONTE-CARLO
    # ======================================================================
    print(f"Mode Monte-Carlo : {N_MC} runs par scénario...\n")

    err_mc   = {}   # nom -> err_all
    Pref_mc  = {}   # nom -> P_ref
    tvref_mc = {}   # nom -> tv_ref
    temps    = None

    for nom, cfg in CONFIGS:
        print(f"  {nom} ...")
        _, err_all, P_ref, tv_ref, temps = run_monte_carlo(
            n_mc=N_MC, base_seed=BASE_SEED_MC, **cfg)
        err_mc[nom]   = err_all
        Pref_mc[nom]  = P_ref
        tvref_mc[nom] = tv_ref

    # --- MSE position drone 2, moyenné sur tous les runs ------------------
    print("\n=== MSE position drone 2 (moyenne sur les runs) ===")
    for nom, cfg in CONFIGS:
        mse_runs = np.mean(err_mc[nom][:, :, 8:10]**2)
        print(f"  {nom:<12} : {mse_runs:.4f}")

    # --- Figures de cohérence (spaghetti + RMSE vs ±3σ) -------------------
    # Recommandation : pour la consistance, on montre TOUS les runs.
    # On affiche les scénarios les plus parlants (A cohérent, C incohérent).
    figure_mc_consistance(err_mc["Scénario A"], Pref_mc["Scénario A"], temps,
                          base=8, nom="Scénario A", drone=2)
    figure_mc_consistance(err_mc["Scénario C"], Pref_mc["Scénario C"], temps,
                          base=8, nom="Scénario C", drone=2)

    # --- Figure de synthèse : RMSE des 4 scénarios ------------------------
    figure_mc_rmse_position([
        (err_mc["Scénario A"], "A — Complet, biais estimé",            'green'),
        (err_mc["Scénario B"], "B — Complet, biais non estimé",         'orange'),
        (err_mc["Scénario C"], "C — Sans distances, biais non estimé",  'red'),
        (err_mc["Scénario D"], "D — Sans distances, biais estimé",      'blue'),
    ], temps, drone=2)

    plt.show()

else:
    # ======================================================================
    #  MODE SINGLE-RUN (comportement d'origine)
    # ======================================================================

    # --- Scénario A : configuration complète, avec biais estimé (référence) ---
    print("Scénario A : tous capteurs, biais estimé...")
    tv, tk_A, Ph_A, temps, gps_A, imu_A, gpsv_A = run_ekf(
        nom_scenario="Scénario A",
        compenser_biais=True, seed=0,
        use_gps=True, use_imu=True, use_distances=True,
        show_corridors=True)

    # --- Scénario B : tous capteurs, sans estimer le biais ---
    print("Scénario B : tous capteurs, biais NON estimé...")
    _, tk_B, Ph_B, _, gps_B, imu_B, gpsv_B = run_ekf(
        nom_scenario="Scénario B",
        compenser_biais=False, seed=0,
        use_gps=True, use_imu=True, use_distances=True,
        show_corridors=True)

    # --- Scénario C : sans distances, sans estimer le biais (dérive parabolique) ---
    print("Scénario C : sans distances, biais NON estimé...")
    _, tk_C, Ph_C, _, gps_C, imu_C, gpsv_C = run_ekf(
        nom_scenario="Scénario C",
        compenser_biais=False, seed=0,
        use_gps=True, use_imu=True, use_distances=False,
        show_corridors=True)

    # --- Scénario D : sans distances, avec biais estimé ---
    print("Scénario D : sans distances, biais estimé...")
    _, tk_D, Ph_D, _, gps_D, imu_D, gpsv_D = run_ekf(
        nom_scenario="Scénario D",
        compenser_biais=True, seed=0,
        use_gps=True, use_imu=True, use_distances=False,
        show_corridors=True)

    # ----------------------------------------------------------------------
    # 8. Figures de présentation
    # ----------------------------------------------------------------------
    mse = lambda a, b: np.square(a - b).mean()
    print("\n=== MSE position drone 2 ===")
    for label, tk in [("A - Complet + biais estimé",          tk_A),
                      ("B - Complet, biais non estimé",        tk_B),
                      ("C - Sans distances, biais non estimé", tk_C),
                      ("D - Sans distances, biais estimé",     tk_D)]:
        print(f"  {label:<40} : {mse(tv[:,8:10], tk[:,8:10]):.4f}")

    # Figure — Comparaison B vs A : distances compensent le biais
    f2 = figure_comparaison_biais(tv, tk_A, tk_B, temps,
                                   label_avec="Biais estimé (A)",
                                   label_sans="Biais non estimé (B)")
    f2.suptitle("Scénario A vs B", fontsize=13, fontweight='bold')

    # Figure — Comparaison C vs D : sans distances, la dérive parabolique apparaît
    f3 = figure_comparaison_biais(tv, tk_D, tk_C, temps,
                                   label_avec="Biais estimé (D)",
                                   label_sans="Biais non estimé (C)")
    f3.suptitle("Scénario C vs D", fontsize=13, fontweight='bold')

    # Figure — 4 scénarios, erreur de position drone 2
    fig4, ax = plt.subplots(figsize=(10, 5))
    for tk, label, color, ls in [
        (tk_A, "A — Complet, biais estimé",           'green',  '-'),
        (tk_B, "B — Complet, biais non estimé",        'orange', '--'),
        (tk_C, "C — Sans distances, biais non estimé", 'red',    '-.'),
        (tk_D, "D — Sans distances, biais estimé",     'blue',   ':'),
    ]:
        err = np.sqrt((tv[:,8]-tk[:,8])**2 + (tv[:,9]-tk[:,9])**2)
        ax.plot(temps, err, color=color, linestyle=ls, lw=1.8, label=label)
    ax.set_title("Comparaison 4 scénarios", fontsize=12)
    ax.set_xlabel("Temps (s)"); ax.set_ylabel("erreur (m)")
    ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.7)
    fig4.tight_layout()

    # Figure — Trajectoires scénario de référence
    f5 = figure_trajectoires(tv, [
        (tk_A, 'Scénario A', 'green', '-'),
        (tk_B, 'Scénario B', 'orange', '--'),
        (tk_C, 'Scénario C', 'red', '-.'),
        (tk_D, 'Scénario D', 'blue', ':'),
    ], temps, gps_A)

    plt.show()
