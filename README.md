"""
============================================================================
 EKF — Localisation coopérative de 3 drones
============================================================================
 État par drone (8 variables) : [x, y, vx, vy, ax, ay, bx, by]
   - position, vitesse, accélération, biais d'accéléromètre
 État global : 24 = 3 x 8

 Capteurs :
   - GPS sur le drone 1               -> (x1, y1)        toutes les dt_capteur
   - Distances inter-drones            -> d12, d23, d13   toutes les dt_capteur
   - IMU (accéléromètre) sur drone 2   -> (ax2+bx2, ay2+by2) toutes les dt_imu

 Le filtre connaît la commande des drones 1 et 3 (coopératifs), mais PAS
 celle du drone 2 : son accélération est estimée via l'IMU + un modèle
 de marche aléatoire (d'où la ligne "accélération constante" dans F_kalman_2).
============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.linalg import block_diag

np.random.seed(0)  # reproductibilité (à retirer pour du vrai aléatoire)

# --------------------------------------------------------------------------
# 1. Paramètres de simulation
# --------------------------------------------------------------------------
t_max      = 16.0
dt         = 0.1
dt_capteur = 0.5      # cadence GPS + distances
dt_imu     = 0.1      # cadence IMU
n_drone    = 3
n_var      = 8        # variables d'état par drone
N          = n_drone * n_var
n_steps    = int(round(t_max / dt))

ratio_imu = int(round(dt_imu / dt))      # un pas IMU tous les `ratio_imu` pas
ratio_gps = int(round(dt_capteur / dt))  # un pas GPS tous les `ratio_gps` pas

# --------------------------------------------------------------------------
# 2. Écarts-types
# --------------------------------------------------------------------------
# Bruit "physique" injecté dans la vérité terrain
sigma_w_accel = 5e-2          # bruit de modèle sur l'accélération
sigma_w_autre = 1e-6          # bruit négligeable sur le reste

# Bruit réel des capteurs (utilisé pour générer les mesures bruitées)
sigma_gps = 0.5
sigma_acc = 0.01
sigma_d   = 0.5

# Bruit supposé par le filtre (matrices R) — volontairement >= bruit réel
sigma_R_gps = 0.5
sigma_R_acc = 0.1
sigma_R_d   = 0.5

# Incertitude initiale (P0)
sigma_P_x, sigma_P_v, sigma_P_a, sigma_P_b = 2.0, 0.5, 0.5, 1.0

# --------------------------------------------------------------------------
# 3. Matrices du modèle
# --------------------------------------------------------------------------
def Bmat(cols):
    """Place la commande (2 composantes) sur les lignes accélération (ax, ay)."""
    M = np.zeros((8, 6))
    M[4, cols[0]] = 1.0
    M[5, cols[1]] = 1.0
    return M

def Fmat(accel_constante):
    """Matrice de transition d'un drone.
    accel_constante=True  -> a_{k+1} = a_k (marche aléatoire, cas drone 2)
    accel_constante=False -> a_{k+1} = 0   (l'accel vient uniquement de B.u)"""
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

# Vérité : les 3 drones reçoivent leur accélération de la commande
F_vrai = block_diag(Fmat(False), Fmat(False), Fmat(False))
B_vrai = np.concatenate((Bmat([0, 1]), Bmat([2, 3]), Bmat([4, 5])), axis=0)

# Filtre : connaît u1 et u3 ; pour le drone 2, accel = marche aléatoire (B=0)
F_kalman = block_diag(Fmat(False), Fmat(True), Fmat(False))
B_kalman = np.concatenate((Bmat([0, 1]), np.zeros((8, 6)), Bmat([4, 5])), axis=0)

# Bruit de processus Q (diagonale)
Q = np.eye(N) * 1e-3
for b in (0, 8, 16):
    Q[b+4, b+4] = Q[b+5, b+5] = 0.5**2     # accel
    Q[b+6, b+6] = Q[b+7, b+7] = 1e-5**2    # biais (quasi figé)
Q[10, 10] = Q[11, 11] = 0.1**2             # vitesse drone 2 (plus libre)
Q[12, 12] = Q[13, 13] = 1e-2**2            # accel drone 2

# Matrices de bruit de mesure
R_gps = np.diag([sigma_R_gps**2, sigma_R_gps**2,
                 sigma_R_d**2,   sigma_R_d**2,   sigma_R_d**2])
R_imu = np.diag([sigma_R_acc**2, sigma_R_acc**2])

I_N = np.eye(N)

# Écarts-types du bruit de vérité, par composante
w_sigma = np.full(N, sigma_w_autre)
for b in (0, 8, 16):
    w_sigma[b+4] = w_sigma[b+5] = sigma_w_accel

# --------------------------------------------------------------------------
# 4. Initialisation des états
# --------------------------------------------------------------------------
X_vrai = np.concatenate(([0, 10, 1, 0, 0, 0, 0,   0],
                         [10, 0, 1, 0, 0, 0, 0.5, -0.2],
                         [0, -10, 1, 0, 0, 0, 0,  0])).astype(float)

# Erreur initiale : on ne se trompe que sur la POSITION de chaque drone
erreur_init = np.zeros(N)
for b in (0, 8, 16):
    erreur_init[b:b+2] = np.random.normal(0, 2.0, size=2)
X_est = X_vrai + erreur_init

# Covariance initiale P0
P_est = np.eye(N)
for b in (0, 8, 16):
    P_est[b,   b]   = P_est[b+1, b+1] = sigma_P_x**2
    P_est[b+2, b+2] = P_est[b+3, b+3] = sigma_P_v**2
    P_est[b+4, b+4] = P_est[b+5, b+5] = sigma_P_a**2
    P_est[b+6, b+6] = P_est[b+7, b+7] = sigma_P_b**2

# --------------------------------------------------------------------------
# 5. Tableaux d'historique (indexés proprement de 0 à n_steps)
# --------------------------------------------------------------------------
traj_vrai   = np.zeros((n_steps + 1, N))
traj_kalman = np.zeros((n_steps + 1, N))
P_hist      = np.zeros((n_steps + 1, N, N))
temps       = np.zeros(n_steps + 1)

traj_vrai[0]   = X_vrai
traj_kalman[0] = X_est
P_hist[0]      = P_est

# Mesures (pour superposition sur les graphes)
mes_gps  = []   # (t, x1, y1)
mes_imu  = []   # (t, ax2_mes, ay2_mes, ax2_vrai, ay2_vrai)
mes_gpsv = []   # (x1_vrai, y1_vrai) au même instant

# --------------------------------------------------------------------------
# 6. Helpers EKF
# --------------------------------------------------------------------------
def maj_kalman(X, P, H, innov, R):
    """Étape de correction (forme de Joseph -> P reste symétrique et définie positive)."""
    S = H @ P @ H.T + R
    K = P @ H.T @ inv(S)
    X = X + K @ innov
    A = I_N - K @ H
    P = A @ P @ A.T + K @ R @ K.T
    return X, P

# --------------------------------------------------------------------------
# 7. Boucle principale
# --------------------------------------------------------------------------
phi_x = phi_y = 0.0

for k in range(1, n_steps + 1):
    step = k - 1            # indice de pas (pour la cadence des capteurs)
    t    = k * dt

    # ---- Commande -------------------------------------------------------
    if step < n_steps / 3:
        u_vrai = np.array([1., 0., 1., 0., 1., 0.])
    elif step < 2 * n_steps / 3:
        phi_x += 5 * dt      # omega_x = 5 rad/s
        phi_y += 1 * dt      # omega_y = 1 rad/s
        u_vrai = np.array([np.cos(phi_x), np.sin(phi_y),
                           np.cos(phi_x), np.sin(phi_y),
                           np.cos(phi_x), np.sin(phi_y)])
    else:
        u_vrai = np.array([1., 0., 1., 0., 1., 0.])

    # Le filtre reçoit une commande bruitée (et ne touche pas u1 en x,y ici)
    err_cmd = np.random.normal(0, 0.1, size=6)
    err_cmd[0:2] = 0.0
    u_kalman = u_vrai + err_cmd

    # ---- Propagation de la vérité --------------------------------------
    X_vrai = F_vrai @ X_vrai + B_vrai @ u_vrai + np.random.normal(0, 1, N) * w_sigma

    # ---- Prédiction du filtre ------------------------------------------
    Xc = F_kalman @ X_est + B_kalman @ u_kalman
    Pc = F_kalman @ P_est @ F_kalman.T + Q

    # ---- Correction IMU (drone 2) — mises à jour SÉQUENTIELLES ----------
    # On corrige directement (Xc, Pc) : la correction n'est JAMAIS jetée.
    if step % ratio_imu == 0:
        mes = np.array([X_vrai[12] + X_vrai[14] + np.random.normal(0, sigma_acc),
                        X_vrai[13] + X_vrai[15] + np.random.normal(0, sigma_acc)])
        H = np.zeros((2, N))
        H[0, 12] = H[0, 14] = 1.0   # mesure = ax2 + bx2
        H[1, 13] = H[1, 15] = 1.0   # mesure = ay2 + by2
        innov = mes - H @ Xc
        Xc, Pc = maj_kalman(Xc, Pc, H, innov, R_imu)
        mes_imu.append((t, mes[0], mes[1], X_vrai[12], X_vrai[13]))

    # ---- Correction GPS + distances ------------------------------------
    if step % ratio_gps == 0:
        # Jacobien linéarisé AUTOUR DE L'ESTIMÉE COURANTE Xc (bonne pratique EKF)
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
        H[0, 0] = 1.0                      # GPS x1
        H[1, 1] = 1.0                      # GPS y1
        H[2, 0] =  (Xc[0]-Xc[8])  / d12;  H[2, 1] =  (Xc[1]-Xc[9])  / d12
        H[2, 8] = -H[2, 0];               H[2, 9] = -H[2, 1]           # d12
        H[3, 8] =  (Xc[8]-Xc[16]) / d23;  H[3, 9] =  (Xc[9]-Xc[17]) / d23
        H[3,16] = -H[3, 8];               H[3,17] = -H[3, 9]           # d23
        H[4, 0] =  (Xc[0]-Xc[16]) / d13;  H[4, 1] =  (Xc[1]-Xc[17]) / d13
        H[4,16] = -H[4, 0];               H[4,17] = -H[4, 1]           # d13

        h_pred = np.array([Xc[0], Xc[1], d12, d23, d13])
        innov  = mes - h_pred
        Xc, Pc = maj_kalman(Xc, Pc, H, innov, R_gps)
        mes_gps.append((t, mes[0], mes[1]))
        mes_gpsv.append((X_vrai[0], X_vrai[1]))

    # ---- Enregistrement -------------------------------------------------
    X_est, P_est = Xc, Pc
    traj_vrai[k]   = X_vrai
    traj_kalman[k] = X_est
    P_hist[k]      = P_est
    temps[k]       = t

# --------------------------------------------------------------------------
# 8. Erreurs (MSE de position)
# --------------------------------------------------------------------------
def mse(a, b): return np.square(a - b).mean()

print("=== MSE position (x, y) ===")
for d, b in [(1, 0), (2, 8), (3, 16)]:
    v = traj_vrai[:, b:b+2]; e = traj_kalman[:, b:b+2]
    print(f"  Drone {d} : globale {mse(v, e):7.4f} | "
          f"initiale {mse(v[0], e[0]):7.4f} | finale {mse(v[-1], e[-1]):7.4f}")

# --------------------------------------------------------------------------
# 9. Affichage
# --------------------------------------------------------------------------
labels = ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'bx', 'by']
mes_gps  = np.array(mes_gps)  if mes_gps  else np.empty((0, 3))
mes_gpsv = np.array(mes_gpsv) if mes_gpsv else np.empty((0, 2))
mes_imu  = np.array(mes_imu)  if mes_imu  else np.empty((0, 5))

def figure_drone(d, base):
    fig, axs = plt.subplots(4, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Analyse EKF — Drone {d}  (erreur d'estimation et couloir ±3σ)",
                 fontsize=13, fontweight='bold')
    axs = axs.flatten()
    for i in range(8):
        idx   = base + i
        sigma = np.sqrt(P_hist[:, idx, idx])
        err   = traj_kalman[:, idx] - traj_vrai[:, idx]
        axs[i].fill_between(temps, -3*sigma, 3*sigma, color='blue', alpha=0.2,
                            label=r'Couloir $\pm 3\sigma$')
        axs[i].plot(temps, err, color='green', label='Erreur EKF')
        axs[i].axhline(0, color='k', lw=0.6)
        axs[i].set_title(f"{labels[i]} : estimé − vrai", fontsize=10)
        axs[i].grid(True, linestyle=':', alpha=0.7)
    # Superposition des mesures GPS pour le drone 1 (x, y)
    if d == 1 and len(mes_gps):
        axs[0].scatter(mes_gps[:, 0], mes_gps[:, 1] - mes_gpsv[:, 0],
                       color='red', marker='x', s=20, label='Mesure GPS')
        axs[1].scatter(mes_gps[:, 0], mes_gps[:, 2] - mes_gpsv[:, 1],
                       color='red', marker='x', s=20, label='Mesure GPS')
    # Superposition des mesures IMU pour le drone 2 (ax, ay)
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

f1 = figure_drone(1, 0)
f2 = figure_drone(2, 8)
f3 = figure_drone(3, 16)

# Trajectoires 2D
f4 = plt.figure(figsize=(8, 7))
i1, i2 = n_steps // 3, 2 * n_steps // 3
if len(mes_gps):
    plt.scatter(mes_gps[:, 1], mes_gps[:, 2], color='red', marker='x',
                s=20, label='Mesures GPS (Drone 1)')
for d, base, mk in [(1, 0, '^'), (2, 8, 'o'), (3, 16, 's')]:
    plt.plot(traj_vrai[:, base], traj_vrai[:, base+1], color='black',
             marker=mk, markevery=[i1], label=f'Drone {d} — vérité')
    plt.plot(traj_kalman[:, base], traj_kalman[:, base+1], color='green',
             linestyle='-', marker=mk, markevery=[i2], label=f'Drone {d} — EKF')
plt.xlabel("X"); plt.ylabel("Y"); plt.title("Trajectoires des 3 drones")
plt.legend(fontsize=8); plt.grid(True); plt.axis('equal')

# Sauvegarde
import os
os.makedirs("/mnt/user-data/outputs", exist_ok=True)
for fig, name in [(f1, "drone1"), (f2, "drone2"), (f3, "drone3"), (f4, "trajectoires")]:
    fig.savefig(f"/mnt/user-data/outputs/ekf_{name}.png", dpi=110, bbox_inches='tight')

plt.show()
