"""
============================================================================
 EKF DISTRIBUÉ — Localisation coopérative de 3 drones
 Chaque drone porte son propre estimateur (état 8 + covariance 8x8).
 Fusion inter-drones par INTERSECTION DE COVARIANCE (CI) sur les distances.
============================================================================
 Différences clés avec le centralisé :
   - Pas de matrice P globale 24x24 : 3 blocs 8x8 indépendants.
   - Les corrélations croisées entre drones sont INCONNUES -> on utilise la
     CI, qui reste cohérente (jamais sur-confiante) sans les connaître.
   - Chaque lien de distance d_ij déclenche DEUX updates CI symétriques :
     une pour i (voisin = j), une pour j (voisin = i). C'est l'équivalent
     distribué du fait qu'au centralisé une ligne de H_dist remplit les
     colonnes des deux drones.

 Notations volontairement proches du centralisé :
   run_ekf(...), traj_vrai, traj_kalman, P_hist, temps,
   flags use_gps / use_imu / use_distances / compenser_biais,
   figure_drone / figure_comparaison_biais / figure_trajectoires.
============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.linalg import block_diag
from scipy.optimize import minimize_scalar

# --------------------------------------------------------------------------
# 1. Paramètres de simulation  (identiques au centralisé)
# --------------------------------------------------------------------------
t_max      = 16.0
dt         = 0.1
dt_capteur = 0.5
dt_imu     = 0.1
n_drone    = 3
n_var      = 8
N          = n_drone * n_var          # 24, utilisé seulement pour la vérité
n_steps    = int(round(t_max / dt))

ratio_imu = int(round(dt_imu / dt))
ratio_gps = int(round(dt_capteur / dt))

# --------------------------------------------------------------------------
# 2. Écarts-types  (identiques au centralisé)
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
# 3. Matrices du modèle  (mêmes briques que le centralisé, version 8x8)
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

# Vérité physique : grand vecteur 24 (comme le centralisé)
F_vrai = block_diag(Fmat(False), Fmat(False), Fmat(False))
B_vrai = np.concatenate((Bmat([0, 1]), Bmat([2, 3]), Bmat([4, 5])), axis=0)

# Côté filtre : chaque drone a SES propres F et B locaux 8x8
# - Drone 1 et 3 : commande connue (B local non nul), accel non constante
# - Drone 2 : pas de commande connue -> B local nul + accel constante (estimée)
def Bmat_local():
    """Commande locale 2D (ax, ay) -> agit sur les composantes accel."""
    M = np.zeros((8, 2))
    M[4, 0] = 1.0
    M[5, 1] = 1.0
    return M

F_kalman_local = {1: Fmat(False), 2: Fmat(True), 3: Fmat(False)}
B_kalman_local = {1: Bmat_local(), 2: np.zeros((8, 2)), 3: Bmat_local()}

I8 = np.eye(8)

# Bruit de modèle sur la vérité (24)
w_sigma = np.full(N, sigma_w_autre)
for b in (0, 8, 16):
    w_sigma[b+4] = w_sigma[b+5] = sigma_w_accel

X_vrai_init = np.concatenate(([0,  10, 1, 0, 0, 0,  0,    0],
                               [10,  0, 1, 0, 0, 0,  0.5, -0.2],
                               [0, -10, 1, 0, 0, 0,  0,    0])).astype(float)

# Bruit de modèle local Q (8x8) — même réglage que les blocs du centralisé
def make_Q_local(drone_id):
    Q = np.eye(8) * 1e-3
    Q[4, 4] = Q[5, 5] = 0.5**2
    Q[6, 6] = Q[7, 7] = 1e-5**2
    if drone_id == 2:                 # le drone 2 a un Q vitesse/accel spécifique
        Q[2, 2] = Q[3, 3] = 0.1**2
        Q[4, 4] = Q[5, 5] = 1e-2**2
    return Q

# --------------------------------------------------------------------------
# 4. Classe estimateur embarqué
# --------------------------------------------------------------------------
class DistributedDrone:
    """Estimateur local d'un drone : état 8, covariance 8x8."""

    def __init__(self, drone_id, x_init, P_init, Q_local, compenser_biais):
        self.id   = drone_id
        self.x    = x_init.copy()
        self.P    = P_init.copy()
        self.Q    = Q_local.copy()
        self.F    = F_kalman_local[drone_id]
        self.B    = B_kalman_local[drone_id]
        self.compenser_biais = compenser_biais
        if not compenser_biais:                 # biais gelé
            self.x[6:8] = 0.0
            self.P[6, 6] = self.P[7, 7] = 1e-8
            self.Q[6, 6] = self.Q[7, 7] = 1e-8

    # ----- Prédiction locale -----
    def predict(self, u_local):
        self.x = self.F @ self.x + self.B @ u_local
        self.P = self.F @ self.P @ self.F.T + self.Q

    # ----- Update capteur local (GPS ou IMU) — forme de Joseph -----
    def update_local(self, mes, H, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ inv(S)
        innov = mes - H @ self.x
        self.x = self.x + K @ innov
        A = I8 - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T

    # ----- Paquet radio minimal à émettre vers les voisins -----
    def get_paquet_radio(self):
        """
        Grandeurs STRICTEMENT nécessaires à un voisin pour une update CI
        sur une mesure de distance : la position (2) + le bloc 2x2 de
        covariance position (3 termes par symétrie) = 5 floats.
        Exact pour une distance (le reste de P est annulé par le jacobien).
        """
        return {
            "pos": self.x[0:2].copy(),          # (x, y)
            "Ppos": self.P[0:2, 0:2].copy(),    # bloc 2x2 symétrique -> 3 floats utiles
        }

    # ----- Fusion CI groupée : toutes les distances du drone en UN update -----
    def update_distances_CI_batch(self, liens, R_scalaire):
        """
        liens : liste de (d_mesure, paquet_voisin).
        On empile toutes les mesures de distance concernant CE drone en un
        seul update CI. La covariance locale n'est dilatée qu'UNE fois (1/omega),
        ce qui évite l'empilement de dilatations qui faisait exploser P.
        Le bloc du voisin (2x2) intervient via R augmenté, projeté par Hj.
        """
        if not liens:
            return
        rows_Hi, innovs, R_diag = [], [], []
        for d_mesure, paquet in liens:
            xi, yi = self.x[0], self.x[1]
            xj, yj = paquet["pos"]
            Pj = paquet["Ppos"]
            d_pred = np.hypot(xi - xj, yi - yj)
            if d_pred < 1e-4:
                d_pred = 1e-4
            Hi = np.zeros(8)
            Hi[0] = (xi - xj) / d_pred
            Hi[1] = (yi - yj) / d_pred
            Hj = np.array([-(xi - xj) / d_pred, -(yi - yj) / d_pred])
            # Incertitude du voisin projetée sur la mesure scalaire
            var_voisin = Hj @ Pj @ Hj.T
            rows_Hi.append(Hi)
            innovs.append(d_mesure - d_pred)
            R_diag.append(R_scalaire + var_voisin)

        H = np.vstack(rows_Hi)                 # (m, 8)
        innov = np.array(innovs)               # (m,)
        Rv = np.diag(R_diag)                   # (m, m) : R + part voisin

        # Une seule dilatation CI sur la covariance locale
        def cout_CI(omega):
            Pi = self.P / omega
            S = H @ Pi @ H.T + Rv / (1.0 - omega)
            K = Pi @ H.T @ inv(S)
            P_up = Pi - K @ H @ Pi
            return np.trace(P_up)

        sol = minimize_scalar(cout_CI, bounds=(0.01, 0.99), method='bounded')
        omega = sol.x
        Pi = self.P / omega
        S = H @ Pi @ H.T + Rv / (1.0 - omega)
        K = Pi @ H.T @ inv(S)
        self.x = self.x + K @ innov
        self.P = Pi - K @ H @ Pi

    # ----- Update distance inter-drone par Intersection de Covariance (unitaire) -----
    def update_distance_CI(self, d_mesure, paquet_voisin, R_dist):
        """
        Recale CE drone à partir d'une mesure de distance vers un voisin,
        en ne connaissant du voisin que (pos, Ppos) — cf. get_paquet_radio.
        La CI fusionne sans connaître la corrélation croisée i<->j.
        """
        xi, yi = self.x[0], self.x[1]
        xj, yj = paquet_voisin["pos"]
        Pj = paquet_voisin["Ppos"]              # 2x2

        d_pred = np.hypot(xi - xj, yi - yj)
        if d_pred < 1e-4:
            d_pred = 1e-4

        # Jacobien côté i (1x8) : seules les colonnes position sont non nulles
        Hi = np.zeros((1, 8))
        Hi[0, 0] = (xi - xj) / d_pred
        Hi[0, 1] = (yi - yj) / d_pred

        # Jacobien côté voisin réduit au bloc position (1x2)
        Hj = np.array([[-(xi - xj) / d_pred, -(yi - yj) / d_pred]])

        innov = d_mesure - d_pred

        # Coût CI : trace de la covariance corrigée en fonction de omega
        def cout_CI(omega):
            Pi = self.P / omega
            Pjs = Pj / (1.0 - omega)
            S = Hi @ Pi @ Hi.T + Hj @ Pjs @ Hj.T + R_dist
            K = Pi @ Hi.T / S[0, 0]
            P_up = Pi - K @ Hi @ Pi
            return np.trace(P_up)

        sol = minimize_scalar(cout_CI, bounds=(0.01, 0.99), method='bounded')
        omega = sol.x

        Pi  = self.P / omega
        Pjs = Pj / (1.0 - omega)
        S   = Hi @ Pi @ Hi.T + Hj @ Pjs @ Hj.T + R_dist
        K   = Pi @ Hi.T / S[0, 0]

        self.x = self.x + (K * innov).flatten()
        self.P = Pi - K @ Hi @ Pi

# --------------------------------------------------------------------------
# 5. Figures utilitaires  (reprises telles quelles du centralisé)
# --------------------------------------------------------------------------
labels = ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'bx', 'by']

def figure_drone(nom_scenario, d, base, tv, tk, P_hist, temps, mes_gps, mes_imu, mes_gpsv, titre_suffix=""):
    fig, axs = plt.subplots(4, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{nom_scenario} - Drone {d}", fontsize=13, fontweight='bold')
    axs = axs.flatten()
    for i in range(8):
        idx   = base + i
        sigma = np.sqrt(P_hist[:, d-1, i, i])     # P_hist distribué : (T, drone, 8, 8)
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
    err_sans = np.sqrt((tv[:,8]-tk_sans[:,8])**2 + (tv[:,9]-tk_sans[:,9])**2)
    axs[1,1].plot(temps, err_avec, 'g',   lw=1.5, label=label_avec)
    axs[1,1].plot(temps, err_sans, 'r--', lw=1.5, label=label_sans)
    axs[1,1].set_title("Erreur de position euclidienne — Drone 2")
    axs[1,1].set_ylabel("||erreur|| (m)"); axs[1,1].set_xlabel("Temps (s)")
    axs[1,1].legend(); axs[1,1].grid(True, linestyle=':', alpha=0.7)

    fig.tight_layout()
    return fig

def figure_trajectoires(tv, scenarios, temps, mes_gps):
    fig = plt.figure(figsize=(10, 8))
    for d, base, mk in [(1, 0, '^'), (2, 8, 'o'), (3, 16, 's')]:
        plt.plot(tv[:, base], tv[:, base+1], color='black', lw=2,
                 label=f'Drone {d} — vérité')
    for tk, label, color, ls in scenarios:
        for d, base, mk in [(1, 0, '^'), (2, 8, 'o'), (3, 16, 's')]:
            lbl = label if d == 1 else '_nolegend'
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
# 6. Fonction principale distribuée
# --------------------------------------------------------------------------
def run_ekf(nom_scenario="", compenser_biais=True, seed=0,
            use_gps=True, use_imu=True, use_distances=True,
            show_corridors=False):
    np.random.seed(seed)

    # ---- Vérité ----------------------------------------------------------
    X_vrai = X_vrai_init.copy()

    # ---- Estimateurs locaux ----------------------------------------------
    erreur_init = np.zeros(N)
    for b in (0, 8, 16):
        erreur_init[b:b+2] = np.random.normal(0, 2.0, size=2)
    X_est0 = X_vrai + erreur_init

    def make_P_local():
        P = np.eye(8)
        P[0, 0] = P[1, 1] = sigma_P_x**2
        P[2, 2] = P[3, 3] = sigma_P_v**2
        P[4, 4] = P[5, 5] = sigma_P_a**2
        P[6, 6] = P[7, 7] = sigma_P_b**2
        return P

    drones = {
        1: DistributedDrone(1, X_est0[0:8],   make_P_local(), make_Q_local(1), compenser_biais),
        2: DistributedDrone(2, X_est0[8:16],  make_P_local(), make_Q_local(2), compenser_biais),
        3: DistributedDrone(3, X_est0[16:24], make_P_local(), make_Q_local(3), compenser_biais),
    }

    # ---- Historique ------------------------------------------------------
    traj_vrai   = np.zeros((n_steps + 1, N))
    traj_kalman = np.zeros((n_steps + 1, N))
    P_hist      = np.zeros((n_steps + 1, 3, 8, 8))   # (T, drone, 8, 8)
    temps       = np.zeros(n_steps + 1)

    traj_vrai[0]   = X_vrai
    traj_kalman[0] = X_est0
    for j, dr in enumerate((drones[1], drones[2], drones[3])):
        P_hist[0, j] = dr.P

    mes_gps  = []
    mes_imu  = []
    mes_gpsv = []

    # ---- Comptabilité bande passante ------------------------------------
    bytes_radio = 0          # paquets minimaux réellement émis (8 octets/float)
    bytes_naif  = 0          # ce que coûterait l'envoi état+covariance complets

    phi_x = phi_y = 0.0

    for k in range(1, n_steps + 1):
        step = k - 1
        t    = k * dt

        # ---- Commande (identique au centralisé) --------------------------
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

        # ---- Propagation de la vérité ------------------------------------
        X_vrai = F_vrai @ X_vrai + B_vrai @ u_vrai + np.random.normal(0, 1, N) * w_sigma

        # ---- Prédiction locale de chaque drone ---------------------------
        # Drone 1 et 3 reçoivent leur commande, drone 2 non (B local nul)
        drones[1].predict(u_kalman[0:2])
        drones[2].predict(np.zeros(2))
        drones[3].predict(u_kalman[4:6])

        # ---- IMU (drone 2) : capteur local -------------------------------
        if use_imu and step % ratio_imu == 0:
            mes = np.array([X_vrai[12] + X_vrai[14] + np.random.normal(0, sigma_acc),
                            X_vrai[13] + X_vrai[15] + np.random.normal(0, sigma_acc)])
            H = np.zeros((2, 8))
            H[0, 4] = H[0, 6] = 1.0          # ax + bx
            H[1, 5] = H[1, 7] = 1.0          # ay + by
            drones[2].update_local(mes, H, np.diag([sigma_R_acc**2]*2))
            mes_imu.append((t, mes[0], mes[1], X_vrai[12], X_vrai[13]))

        # ---- GPS (drone 1) + distances inter-drones ----------------------
        if step % ratio_gps == 0:

            # GPS : capteur local du drone 1
            if use_gps:
                z = np.array([X_vrai[0] + np.random.normal(0, sigma_gps),
                              X_vrai[1] + np.random.normal(0, sigma_gps)])
                H = np.zeros((2, 8))
                H[0, 0] = 1.0
                H[1, 1] = 1.0
                drones[1].update_local(z, H, np.diag([sigma_R_gps**2]*2))
                mes_gps.append((t, z[0], z[1]))
                mes_gpsv.append((X_vrai[0], X_vrai[1]))

            # Distances : fusion collaborative par CI
            if use_distances:
                # Vraies distances bruitées (mesures physiques partagées)
                d12v = np.hypot(X_vrai[0]-X_vrai[8],  X_vrai[1]-X_vrai[9])
                d23v = np.hypot(X_vrai[8]-X_vrai[16], X_vrai[9]-X_vrai[17])
                d13v = np.hypot(X_vrai[0]-X_vrai[16], X_vrai[1]-X_vrai[17])
                z12 = d12v + np.random.normal(0, sigma_d)
                z23 = d23v + np.random.normal(0, sigma_d)
                z13 = d13v + np.random.normal(0, sigma_d)

                # Chaque drone échange son paquet radio minimal
                paquets = {i: drones[i].get_paquet_radio() for i in (1, 2, 3)}

                # Comptabilité : 5 floats émis par paquet vs 8+36 au naïf
                for _ in (1, 2, 3):
                    bytes_radio += (2 + 3) * 8       # pos(2) + Ppos sym(3)
                    bytes_naif  += (8 + 36) * 8      # état(8) + P sym(36)

                # --- Chaque drone fusionne EN UN SEUL UPDATE toutes les
                #     distances qui le concernent (évite l'empilement des
                #     dilatations CI qui faisait exploser la covariance).
                #     Le drone 2 utilise enfin d23 (c'était l'oubli initial).
                drones[1].update_distances_CI_batch(
                    [(z12, paquets[2]), (z13, paquets[3])], sigma_R_d**2)
                drones[2].update_distances_CI_batch(
                    [(z12, paquets[1]), (z23, paquets[3])], sigma_R_d**2)
                drones[3].update_distances_CI_batch(
                    [(z23, paquets[2]), (z13, paquets[1])], sigma_R_d**2)

        # ---- Enregistrement ---------------------------------------------
        traj_vrai[k]   = X_vrai
        traj_kalman[k] = np.concatenate((drones[1].x, drones[2].x, drones[3].x))
        for j, dr in enumerate((drones[1], drones[2], drones[3])):
            P_hist[k, j] = dr.P
        temps[k] = t

    mes_gps  = np.array(mes_gps)  if mes_gps  else np.empty((0, 3))
    mes_imu  = np.array(mes_imu)  if mes_imu  else np.empty((0, 5))
    mes_gpsv = np.array(mes_gpsv) if mes_gpsv else np.empty((0, 2))

    if show_corridors:
        for d, base in [(1, 0), (2, 8), (3, 16)]:
            figure_drone(nom_scenario, d, base, traj_vrai, traj_kalman, P_hist, temps,
                         mes_gps, mes_imu, mes_gpsv)

    # Stats bande passante stockées en attribut de fonction (accès post-run)
    run_ekf.dernier_bilan_radio = (bytes_radio, bytes_naif)

    return traj_vrai, traj_kalman, P_hist, temps, mes_gps, mes_imu, mes_gpsv

# --------------------------------------------------------------------------
# 7. Scénarios de présentation  (mêmes que le centralisé)
# --------------------------------------------------------------------------
print("Scénario A : tous capteurs, biais estimé...")
tv, tk_A, Ph_A, temps, gps_A, imu_A, gpsv_A = run_ekf(
    nom_scenario="Scénario A (distribué)",
    compenser_biais=True, seed=0,
    use_gps=True, use_imu=True, use_distances=True,
    show_corridors=True)
radio_A = run_ekf.dernier_bilan_radio

print("Scénario B : tous capteurs, biais NON estimé...")
_, tk_B, Ph_B, _, gps_B, imu_B, gpsv_B = run_ekf(
    nom_scenario="Scénario B (distribué)",
    compenser_biais=False, seed=0,
    use_gps=True, use_imu=True, use_distances=True,
    show_corridors=False)

print("Scénario C : sans distances, biais NON estimé...")
_, tk_C, Ph_C, _, gps_C, imu_C, gpsv_C = run_ekf(
    nom_scenario="Scénario C (distribué)",
    compenser_biais=False, seed=0,
    use_gps=True, use_imu=True, use_distances=False,
    show_corridors=False)

print("Scénario D : sans distances, biais estimé...")
_, tk_D, Ph_D, _, gps_D, imu_D, gpsv_D = run_ekf(
    nom_scenario="Scénario D (distribué)",
    compenser_biais=True, seed=0,
    use_gps=True, use_imu=True, use_distances=False,
    show_corridors=False)

# --------------------------------------------------------------------------
# 8. Bilans chiffrés
# --------------------------------------------------------------------------
mse = lambda a, b: np.square(a - b).mean()
print("\n=== MSE position drone 2 ===")
for label, tk in [("A - Complet + biais estimé",          tk_A),
                  ("B - Complet, biais non estimé",        tk_B),
                  ("C - Sans distances, biais non estimé", tk_C),
                  ("D - Sans distances, biais estimé",     tk_D)]:
    print(f"  {label:<40} : {mse(tv[:,8:10], tk[:,8:10]):.4f}")

br, bn = radio_A
print("\n=== Bande passante inter-drones (scénario A) ===")
print(f"  Paquet minimal émis  : {br:>8d} octets")
print(f"  Équivalent naïf      : {bn:>8d} octets")
print(f"  Économie             : {100*(1-br/bn):.1f} %")

# --------------------------------------------------------------------------
# 9. Figures de présentation  (mêmes que le centralisé)
# --------------------------------------------------------------------------
f2 = figure_comparaison_biais(tv, tk_A, tk_B, temps,
                               label_avec="Biais estimé (A)",
                               label_sans="Biais non estimé (B)")
f2.suptitle("Scénario A vs B (distribué)", fontsize=13, fontweight='bold')

f3 = figure_comparaison_biais(tv, tk_D, tk_C, temps,
                               label_avec="Biais estimé (D)",
                               label_sans="Biais non estimé (C)")
f3.suptitle("Scénario C vs D (distribué)", fontsize=13, fontweight='bold')

fig4, ax = plt.subplots(figsize=(10, 5))
for tk, label, color, ls in [
    (tk_A, "A — Complet, biais estimé",            'green',  '-'),
    (tk_B, "B — Complet, biais non estimé",         'orange', '--'),
    (tk_C, "C — Sans distances, biais non estimé",  'red',    '-.'),
    (tk_D, "D — Sans distances, biais estimé",      'blue',   ':'),
]:
    err = np.sqrt((tv[:,8]-tk[:,8])**2 + (tv[:,9]-tk[:,9])**2)
    ax.plot(temps, err, color=color, linestyle=ls, lw=1.8, label=label)
ax.set_title("Comparaison 4 scénarios (distribué)", fontsize=12)
ax.set_xlabel("Temps (s)"); ax.set_ylabel("erreur (m)")
ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.7)
fig4.tight_layout()

f5 = figure_trajectoires(tv, [
    (tk_A, 'Scénario A', 'green', '-'),
    (tk_B, 'Scénario B', 'orange', '--'),
    (tk_C, 'Scénario C', 'red', '-.'),
    (tk_D, 'Scénario D', 'blue', ':'),
], temps, gps_A)

plt.show()
