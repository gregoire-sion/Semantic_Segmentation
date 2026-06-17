"""
============================================================================
 EKF — Localisation coopérative de 3 drones
 + Démonstration de l'importance de l'estimation du biais
============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.linalg import block_diag
import os

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
# 3. Matrices du modèle (définies une seule fois, hors de la fonction)
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

R_gps = np.diag([sigma_R_gps**2, sigma_R_gps**2,
                 sigma_R_d**2,   sigma_R_d**2,   sigma_R_d**2])
R_imu = np.diag([sigma_R_acc**2, sigma_R_acc**2])
I_N   = np.eye(N)

w_sigma = np.full(N, sigma_w_autre)
for b in (0, 8, 16):
    w_sigma[b+4] = w_sigma[b+5] = sigma_w_accel

X_vrai_init = np.concatenate(([0, 10, 1, 0, 0, 0, 0,    0],
                               [10, 0, 1, 0, 0, 0, 0.5, -0.2],
                               [0, -10, 1, 0, 0, 0, 0,   0])).astype(float)

# --------------------------------------------------------------------------
# 4. Helper EKF
# --------------------------------------------------------------------------
def maj_kalman(X, P, H, innov, R):
    """Forme de Joseph — garantit symétrie et définie-positivité de P."""
    S = H @ P @ H.T + R
    K = P @ H.T @ inv(S)
    X = X + K @ innov
    A = I_N - K @ H
    P = A @ P @ A.T + K @ R @ K.T
    return X, P

# --------------------------------------------------------------------------
# 5. Fonction principale — seed fixe pour comparer les deux cas équitablement
# --------------------------------------------------------------------------
def run_ekf(compenser_biais=True, seed=0):
    np.random.seed(seed)

    # ---- Initialisation des états ----------------------------------------
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

    # ---- Gel du biais si compenser_biais=False ---------------------------
    if not compenser_biais:
        X_est[6:8]   = 0.0   # biais drone 1 forcé à 0
        X_est[14:16] = 0.0   # biais drone 2 forcé à 0
        X_est[22:24] = 0.0   # biais drone 3 forcé à 0
        for i in (6, 7, 14, 15, 22, 23):
            P_est[i, i] = 1e-8   # filtre refuse d'apprendre
            Q[i, i]     = 1e-8   # filtre refuse de douter

    # ---- Tableaux d'historique -------------------------------------------
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

    # ---- Boucle principale (inchangée) -----------------------------------
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

        # Correction IMU (drone 2)
        if step % ratio_imu == 0:
            mes = np.array([X_vrai[12] + X_vrai[14] + np.random.normal(0, sigma_acc),
                            X_vrai[13] + X_vrai[15] + np.random.normal(0, sigma_acc)])
            H = np.zeros((2, N))
            H[0, 12] = H[0, 14] = 1.0
            H[1, 13] = H[1, 15] = 1.0
            innov = mes - H @ Xc
            Xc, Pc = maj_kalman(Xc, Pc, H, innov, R_imu)
            mes_imu.append((t, mes[0], mes[1], X_vrai[12], X_vrai[13]))

        # Correction GPS + distances
        if step % ratio_gps == 0:
            d12 = np.hypot(Xc[0]-Xc[8],  Xc[1]-Xc[9])
            d23 = np.hypot(Xc[8]-Xc[16], Xc[9]-Xc[17])
            d13 = np.hypot(Xc[0]-Xc[16], Xc[1]-Xc[17])

            d12v = np.hypot(X_vrai[0]-X_vrai[8],  X_vrai[1]-X_vrai[9])
            d23v = np.hypot(X_vrai[8]-X_vrai[16], X_vrai[9]-X_vrai[17])
            d13v = np.hypot(X_vrai[0]-X_vrai[16], X_vrai[1]-X_vrai[17])

            mes = np.array([X_vrai[0] + np.random.normal(0, sigma_gps),
                            X_vrai[1] + np.random.normal(0, sigma_gps),
                            d12v + np.random.normal(0, sigma_d),
                            d23v + np.random.normal(0, sigma_d),
                            d13v + np.random.normal(0, sigma_d)])

            H = np.zeros((5, N))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            H[2, 0] =  (Xc[0]-Xc[8])  / d12;  H[2, 1] =  (Xc[1]-Xc[9])  / d12
            H[2, 8] = -H[2, 0];               H[2, 9] = -H[2, 1]
            H[3, 8] =  (Xc[8]-Xc[16]) / d23;  H[3, 9] =  (Xc[9]-Xc[17]) / d23
            H[3,16] = -H[3, 8];               H[3,17] = -H[3, 9]
            H[4, 0] =  (Xc[0]-Xc[16]) / d13;  H[4, 1] =  (Xc[1]-Xc[17]) / d13
            H[4,16] = -H[4, 0];               H[4,17] = -H[4, 1]

            h_pred = np.array([Xc[0], Xc[1], d12, d23, d13])
            innov  = mes - h_pred
            Xc, Pc = maj_kalman(Xc, Pc, H, innov, R_gps)
            mes_gps.append((t, mes[0], mes[1]))
            mes_gpsv.append((X_vrai[0], X_vrai[1]))

        # Enregistrement
        X_est, P_est = Xc, Pc
        traj_vrai[k]   = X_vrai
        traj_kalman[k] = X_est
        P_hist[k]      = P_est
        temps[k]       = t

    mes_gps  = np.array(mes_gps)  if mes_gps  else np.empty((0, 3))
    mes_imu  = np.array(mes_imu)  if mes_imu  else np.empty((0, 5))
    mes_gpsv = np.array(mes_gpsv) if mes_gpsv else np.empty((0, 2))

    return traj_vrai, traj_kalman, P_hist, temps, mes_gps, mes_imu, mes_gpsv

# --------------------------------------------------------------------------
# 6. Double exécution (même seed → même trajectoire vraie, même bruit)
# --------------------------------------------------------------------------
print("Run 1 : avec estimation du biais...")
tv, tk_avec, Ph_avec, temps, mes_gps, mes_imu, mes_gpsv = run_ekf(compenser_biais=True,  seed=0)

print("Run 2 : sans estimation du biais...")
_ , tk_sans, Ph_sans, _,     _,       _,       _        = run_ekf(compenser_biais=False, seed=0)

# --------------------------------------------------------------------------
# 7. MSE
# --------------------------------------------------------------------------
def mse(a, b): return np.square(a - b).mean()

print("\n=== MSE position (x, y) ===")
print(f"{'Drone':<8} {'Avec biais':>12} {'Sans biais':>12}")
for d, b in [(1, 0), (2, 8), (3, 16)]:
    v = tv[:, b:b+2]
    print(f"  Drone {d}   {mse(v, tk_avec[:,b:b+2]):>12.4f} {mse(v, tk_sans[:,b:b+2]):>12.4f}")

# --------------------------------------------------------------------------
# 8. Figures d'analyse individuelles (version avec biais — inchangées)
# --------------------------------------------------------------------------
labels = ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'bx', 'by']

def figure_drone(d, base, traj_kalman, P_hist):
    fig, axs = plt.subplots(4, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Analyse EKF — Drone {d}  (erreur d'estimation et couloir ±3σ)",
                 fontsize=13, fontweight='bold')
    axs = axs.flatten()
    for i in range(8):
        idx   = base + i
        sigma = np.sqrt(P_hist[:, idx, idx])
        err   = traj_kalman[:, idx] - tv[:, idx]
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

f1 = figure_drone(1, 0,  tk_avec, Ph_avec)
f2 = figure_drone(2, 8,  tk_avec, Ph_avec)
f3 = figure_drone(3, 16, tk_avec, Ph_avec)

# Trajectoires 2D
f4 = plt.figure(figsize=(8, 7))
i1, i2 = n_steps // 3, 2 * n_steps // 3
if len(mes_gps):
    plt.scatter(mes_gps[:, 1], mes_gps[:, 2], color='red', marker='x',
                s=20, label='Mesures GPS (Drone 1)')
for d, base, mk in [(1, 0, '^'), (2, 8, 'o'), (3, 16, 's')]:
    plt.plot(tv[:, base],      tv[:, base+1],      color='black', marker=mk,
             markevery=[i1], label=f'Drone {d} — vérité')
    plt.plot(tk_avec[:, base], tk_avec[:, base+1], color='green', linestyle='-',
             marker=mk, markevery=[i2], label=f'Drone {d} — EKF')
plt.xlabel("X"); plt.ylabel("Y"); plt.title("Trajectoires des 3 drones")
plt.legend(fontsize=8); plt.grid(True); plt.axis('equal')

# --------------------------------------------------------------------------
# 9. Figure de comparaison biais — centrée sur le drone 2
# --------------------------------------------------------------------------
f5, axs = plt.subplots(2, 2, figsize=(13, 7))
f5.suptitle("Impact de l'estimation du biais — Drone 2  (bx=0.5, by=-0.2)",
            fontsize=13, fontweight='bold')

# Position x
axs[0,0].plot(temps, tv[:,8],       'k',   lw=1.5, label='Vérité')
axs[0,0].plot(temps, tk_avec[:,8],  'g',   lw=1.5, label='Avec biais estimé')
axs[0,0].plot(temps, tk_sans[:,8],  'r--', lw=1.5, label='Sans biais estimé')
axs[0,0].set_title("Position x — Drone 2"); axs[0,0].set_ylabel("x (m)")
axs[0,0].legend(); axs[0,0].grid(True, linestyle=':', alpha=0.7)

# Position y
axs[0,1].plot(temps, tv[:,9],       'k',   lw=1.5)
axs[0,1].plot(temps, tk_avec[:,9],  'g',   lw=1.5)
axs[0,1].plot(temps, tk_sans[:,9],  'r--', lw=1.5)
axs[0,1].set_title("Position y — Drone 2"); axs[0,1].set_ylabel("y (m)")
axs[0,1].grid(True, linestyle=':', alpha=0.7)

# Biais bx estimé vs vrai
axs[1,0].axhline(0.5,  color='k',   lw=1.5, label='Vrai biais bx = 0.5')
axs[1,0].plot(temps, tk_avec[:,14], 'g',   lw=1.5, label='Estimé (avec)')
axs[1,0].plot(temps, tk_sans[:,14], 'r--', lw=1.5, label='Estimé (sans) = 0 figé')
axs[1,0].set_title("Estimation du biais bx — Drone 2"); axs[1,0].set_ylabel("bx")
axs[1,0].set_xlabel("Temps (s)")
axs[1,0].legend(); axs[1,0].grid(True, linestyle=':', alpha=0.7)

# Erreur de position cumulée
err_avec = np.sqrt((tv[:,8]-tk_avec[:,8])**2 + (tv[:,9]-tk_avec[:,9])**2)
err_sans  = np.sqrt((tv[:,8]-tk_sans[:,8])**2 + (tv[:,9]-tk_sans[:,9])**2)
axs[1,1].plot(temps, err_avec, 'g',   lw=1.5, label='Avec biais estimé')
axs[1,1].plot(temps, err_sans,  'r--', lw=1.5, label='Sans biais estimé')
axs[1,1].set_title("Erreur de position euclidienne — Drone 2")
axs[1,1].set_ylabel("||erreur|| (m)"); axs[1,1].set_xlabel("Temps (s)")
axs[1,1].legend(); axs[1,1].grid(True, linestyle=':', alpha=0.7)

f5.tight_layout()

# --------------------------------------------------------------------------
# 10. Sauvegarde
# --------------------------------------------------------------------------
os.makedirs("/mnt/user-data/outputs", exist_ok=True)
for fig, name in [(f1, "drone1"), (f2, "drone2"), (f3, "drone3"),
                  (f4, "trajectoires"), (f5, "comparaison_biais")]:
    fig.savefig(f"/mnt/user-data/outputs/ekf_{name}.png", dpi=110, bbox_inches='tight')

plt.show()
