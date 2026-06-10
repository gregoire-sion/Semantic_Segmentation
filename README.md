
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.optimize import minimize_scalar

# =====================================================================
# 1. CLASSE DU FILTRE DE KALMAN DISTRIBUÉ (L'ESTIMATEUR EMBARQUÉ)
# =====================================================================
class DistributedDrone:
    def __init__(self, drone_id, x_init, P_init, Q_local):
        self.id = drone_id
        self.x_est = x_init.copy()  # [x, y, vx, vy, ax, ay, bx, by]
        self.P_est = P_init.copy()
        self.Q = Q_local.copy()
        
        # Matrice de commande locale (B_local) : 2 entrées (ax_cmd, ay_cmd) vers l'état
        self.B = np.zeros((8, 2))
        self.B[4, 0] = 1.0
        self.B[5, 1] = 1.0
        
        # Matrice de transition cinématique locale
        self.F = np.eye(8)

    def predict(self, u_local, dt):
        """ Étape de propagation locale (Inertie) """
        self.F[0, 2] = dt; self.F[0, 4] = 0.5 * dt**2
        self.F[1, 3] = dt; self.F[1, 5] = 0.5 * dt**2
        self.F[2, 4] = dt
        self.F[3, 5] = dt
        
        self.x_est = self.F @ self.x_est + self.B @ u_local
        self.P_est = self.F @ self.P_est @ self.F.T + self.Q

    def update_local_sensor(self, z_local, sensor_type, R_local):
        """ Étape de mise à jour classique pour capteurs directements branchés """
        I = np.eye(8)
        if sensor_type == 'GPS':
            H = np.zeros((2, 8))
            H[0, 0] = 1.0; H[1, 1] = 1.0
            innov = z_local - H @ self.x_est
        elif sensor_type == 'IMU':
            H = np.zeros((2, 8))
            H[0, 4] = 1.0; H[0, 6] = 1.0  # ax + bx
            H[1, 5] = 1.0; H[1, 7] = 1.0  # ay + by
            z_pred = np.array([self.x_est[4] + self.x_est[6], self.x_est[5] + self.x_est[7]])
            innov = z_local - z_pred
        else:
            return

        S = H @ self.P_est @ H.T + R_local
        K = self.P_est @ H.T @ np.linalg.inv(S)
        self.x_est = self.x_est + K @ innov
        self.P_est = (I - K @ H) @ self.P_est

    def update_inter_drone_distance(self, d_measured, neighbor_x, neighbor_P, R_dist):
        """ Mise à jour distribuée par INTERSECTION DE COVARIANCE (CI) """
        xi, yi = self.x_est[0], self.x_est[1]
        xj, yj = neighbor_x[0], neighbor_x[1]
        
        d_pred = np.sqrt((xi - xj)**2 + (yi - yj)**2)
        if d_pred < 1e-4: d_pred = 1e-4
        
        H_i = np.zeros((1, 8))
        H_i[0, 0] = (xi - xj) / d_pred
        H_i[0, 1] = (yi - yj) / d_pred
        
        H_j = np.zeros((1, 8))
        H_j[0, 0] = -(xi - xj) / d_pred
        H_j[0, 1] = -(yi - yj) / d_pred
        
        innov = d_measured - d_pred

        # Optimisation de la combinaison convexe omega pour minimiser la trace de P
        def covariance_intersection_cost(omega):
            P_scaled_i = self.P_est / omega
            P_scaled_j = neighbor_P / (1.0 - omega)
            S = H_i @ P_scaled_i @ H_i.T + H_j @ P_scaled_j @ H_j.T + R_dist
            K = P_scaled_i @ H_i.T / S[0, 0]
            P_up = P_scaled_i - K @ H_i @ P_scaled_i
            return np.trace(P_up)

        sol = minimize_scalar(covariance_intersection_cost, bounds=(0.01, 0.99), method='bounded')
        omega_opt = sol.x

        P_scaled_i = self.P_est / omega_opt
        P_scaled_j = neighbor_P / (1.0 - omega_opt)
        
        S_opt = H_i @ P_scaled_i @ H_i.T + H_j @ P_scaled_j @ H_j.T + R_dist
        K_opt = P_scaled_i @ H_i.T / S_opt[0, 0]
        
        self.x_est = self.x_est + (K_opt * innov).flatten()
        self.P_est = P_scaled_i - K_opt @ H_i @ P_scaled_i


# =====================================================================
# 2. INITIALISATION DU MONDE PHYSIQUE (COHÉRENT AVEC TON CENTRALISÉ)
# =====================================================================
t_max = 16
dt = 0.1
dt_capteur = 0.5  # Corrigé (évite l'asphyxie à 16s !)
n_steps = int(t_max / dt)

# Sigmas physiques de ta simulation
sigma_gps_1 = 0.1; sigma_acc_2 = 0.5
sigma_dist_12 = 0.1; sigma_dist_23 = 0.1; sigma_dist_13 = 0.1
sigma_ax_phys = 0.1; sigma_ay_phys = 0.1; sigma_bx_phys = 0.000001; sigma_by_phys = 0.000001

# Construction des matrices globales de la VÉRITÉ PHYSIQUE
F_vrai_block = np.array([
    [1, 0, dt, 0, 0.5*dt*dt, 0, 0, 0],
    [0, 1, 0, dt, 0, 0.5*dt*dt, 0, 0],
    [0, 0, 1, 0, dt, 0, 0, 0],
    [0, 0, 0, 1, 0, dt, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]])
F_vrai = block_diag(F_vrai_block, F_vrai_block, F_vrai_block)

B_vrai_block = np.zeros((8, 2))
B_vrai_block[4, 0] = 1.0; B_vrai_block[5, 1] = 1.0
B_vrai = block_diag(B_vrai_block, B_vrai_block, B_vrai_block)

# États initiaux vrais
X_vrai_1 = np.array([0.0, 10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
X_vrai_2 = np.array([10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, -0.2])
X_vrai_3 = np.array([0.0, -10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
X_vrai = np.concatenate((X_vrai_1, X_vrai_2, X_vrai_3))

# =====================================================================
# 3. INITIALISATION DES ESTIMATEURS DISTRIBUÉS
# =====================================================================
sigma_init_large = 0.5  # Permet au filtre de ne pas être trop rigide au départ
P_init_local = np.eye(8) * (sigma_init_large**2)
P_init_local[6, 6] = 1.0  # On donne de l'incertitude initiale au biais du Drone 2 !
P_init_local[7, 7] = 1.0

Q_local = np.diag([0.01, 0.01, 0.01, 0.01, sigma_ax_phys**2, sigma_ay_phys**2, sigma_bx_phys**2, sigma_by_phys**2])

# Triche initiale légère (Test 1 validé ensemble)
erreur_init = np.random.normal(0, 0.01, size=24)
X_est_global = X_vrai.copy() + erreur_init

drone1 = DistributedDrone(1, X_est_global[0:8], P_init_local, Q_local)
drone2 = DistributedDrone(2, X_est_global[8:16], P_init_local, Q_local)
drone3 = DistributedDrone(3, X_est_global[16:24], P_init_local, Q_local)

# Tableaux d'historique pour les graphiques
hist_vrai = np.zeros((n_steps, 24))
hist_est = np.zeros((n_steps, 24))
temps = []

phi_x, phi_y = 0.0, 0.0
Ax1, Ay1, Ax2, Ay2, Ax3, Ay3 = 0.0, 2.0, 0.0, 2.0, 0.0, 2.0

# =====================================================================
# 4. BOUCLE DE SIMULATION TEMPORELLE
# =====================================================================
for step in range(n_steps):
    t = step * dt
    temps.append(t)
    
    # --- Génération de la commande triphasée (Ton code exact) ---
    if step < (n_steps / 3):
        u_vrai = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    elif (n_steps / 3) <= step < (2 * n_steps / 3):
        omega_x, omega_y = 5.0, 1.0
        phi_x += omega_x * dt
        phi_y += omega_y * dt
        u_vrai = np.array([Ax1*np.cos(phi_x), Ay1*np.sin(phi_y), Ax2*np.cos(phi_x), Ay2*np.sin(phi_y), Ax3*np.cos(phi_x), Ay3*np.sin(phi_y)])
    else:
        u_vrai = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    # Extraction des commandes locales pour chaque drone
    u_d1 = u_vrai[0:2]; u_d2 = u_vrai[2:4]; u_d3 = u_vrai[4:6]

    # --- 4.1 PROPAGATION DU MONDE PHYSIQUE (La Nature) ---
    w_vrai = np.zeros(24)
    # Bruit sur les accélérations réelles
    w_vrai[[4, 5, 12, 13, 20, 21]] = np.random.normal(0, sigma_ax_phys, size=6)
    X_vrai = F_vrai @ X_vrai + B_vrai @ u_vrai + w_vrai

    # --- 4.2 ÉTAPE DE PRÉDICTION LOCALE (Les calculateurs embarqués) ---
    drone1.predict(u_d1, dt)
    drone2.predict(u_d2, dt)
    drone3.predict(u_d3, dt)

    # --- 4.3 ÉTAPE DE CORRECTION (Capteurs périodiques) ---
    if step % int(dt_capteur / dt) == 0:
        # Génération des vraies distances géométriques (Formules corrigées !)
        d12_vrai = np.sqrt((X_vrai[0]-X_vrai[8])**2 + (X_vrai[1]-X_vrai[9])**2)
        d23_vrai = np.sqrt((X_vrai[8]-X_vrai[16])**2 + (X_vrai[9]-X_vrai[17])**2)
        d13_vrai = np.sqrt((X_vrai[0]-X_vrai[16])**2 + (X_vrai[1]-X_vrai[17])**2) # Corrigé X_vrai[2] -> [1]

        # Simulation des mesures physiques bruitées
        z_gps_1 = np.array([X_vrai[0], X_vrai[1]]) + np.random.normal(0, sigma_gps_1, size=2)
        z_imu_2 = np.array([X_vrai[12] + X_vrai[14], X_vrai[13] + X_vrai[15]]) + np.random.normal(0, sigma_acc_2, size=2) # Corrigé [12]->[13]
        z_d12 = d12_vrai + np.random.normal(0, sigma_dist_12)
        z_d23 = d23_vrai + np.random.normal(0, sigma_dist_23)
        z_d13 = d13_vrai + np.random.normal(0, sigma_dist_13)

        # -- CORRECTION DES CAPTEURS INTERNES DIRECTS --
        drone1.update_local_sensor(z_gps_1, 'GPS', np.eye(2)*sigma_gps_1**2)
        drone2.update_local_sensor(z_imu_2, 'IMU', np.eye(2)*sigma_acc_2**2)

        # -- FUSION COLLABORATIVE VIA INTERSECTION DE COVARIANCE (Liaison Radio) --
        # Le Drone 2 se recale par rapport au Drone 1 via d12
        drone2.update_inter_drone_distance(z_d12, drone1.x_est, drone1.P_est, np.array([[sigma_dist_12**2]]))
        # Le Drone 3 se recale par rapport au Drone 2 via d23
        drone3.update_inter_drone_distance(z_d23, drone2.x_est, drone2.P_est, np.array([[sigma_dist_23**2]]))
        # Le Drone 3 se recale par rapport au Drone 1 via d13
        drone3.update_inter_drone_distance(z_d13, drone1.x_est, drone1.P_est, np.array([[sigma_dist_13**2]]))

    # Sauvegarde des données synchronisées
    hist_vrai[step] = X_vrai
    hist_est[step] = np.concatenate((drone1.x_est, drone2.x_est, drone3.x_est))

# =====================================================================
# 5. AFFICHAGE DES TRAJECTOIRE (VÉRIFICATION SOUHAITÉE)
# =====================================================================
plt.figure(figsize=(9, 7))
plt.plot(hist_vrai[:, 0], hist_vrai[:, 1], 'k-', label='Drone 1 Vrai')
plt.plot(hist_est[:, 0], hist_est[:, 1], 'g--', label='Drone 1 Estime (GPS)')

plt.plot(hist_vrai[:, 8], hist_vrai[:, 9], 'k-', label='Drone 2 Vrai')
plt.plot(hist_est[:, 8], hist_est[:, 9], 'b--', label='Drone 2 Estime (IMU + Télémétrie CI)')

plt.plot(hist_vrai[:, 16], hist_vrai[:, 17], 'k-', label='Drone 3 Vrai')
plt.plot(hist_est[:, 16], hist_est[:, 17], 'r--', label='Drone 3 Estime (Télémétrie Pure CI)')

plt.title("Navigation Collaborative Distribuée - EKF Covariance Intersection")
plt.xlabel("X (m)"); plt.ylabel("Y (m)"); plt.grid(True); plt.legend()
plt.savefig("trajectoire_distribuee.png")
