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
        
        self.B = np.zeros((8, 2))
        self.B[4, 0] = 1.0
        self.B[5, 1] = 1.0
        

        self.F = np.eye(8)
        self.F[4,4] = 0.0
        self.F[5,5] = 0.0


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


# ===================================
# 2. INITIALISATION DU MONDE PHYSIQUE
# ===================================
t_max = 16
dt = 0.1
dt_capteur = 0.5 
n_steps = int(t_max / dt)
n_steps_capteur = int(t_max/dt_capteur)

# Sigmas physiques de ta simulation


############################################################
#------------Pour bruiter les mesures------------
sigma_gps_1 = 0.5; sigma_acc_2 = 0.1
sigma_dist_12 = 0.1; sigma_dist_23 = 0.1; sigma_dist_13 = 0.1

#----------------------Pour R--------------------
sigma_R_gps_1 = 0.5; sigma_R_acc_2 = 0.1
sigma_R_dist_12 = 0.1; sigma_R_dist_23 = 0.1; sigma_R_dist_13 = 0.1
############################################################

############################################################
#------------Pour bruiter le modèle--------------
sigma_x_phys = 1e-6; sigma_y_phys = 1e-6
sigma_vx_phys = 1e-6; sigma_vy_phys = 1e-6
sigma_ax_phys = 5e-2; sigma_ay_phys = 5e-2
sigma_bx_phys = 1e-6; sigma_by_phys = 1e-6

#---------------------Pour Q---------------------
sigma_Q_x_phys = 1e-1; sigma_Q_y_phys = 1e-1
sigma_Q_vx_phys = 1e-1; sigma_Q_vy_phys = 1e-1
sigma_Q_ax_phys = 5e-2; sigma_Q_ay_phys = 5e-2
sigma_Q_bx_phys = 1e-5; sigma_Q_by_phys = 1e-5
############################################################

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
sigma_init_large = 5 
P_init_local = np.eye(8) * (sigma_init_large**2)
P_init_local[6, 6] = 1.0
P_init_local[7, 7] = 1.0

Q_local = np.diag([sigma_Q_x_phys**2, sigma_Q_y_phys**2, sigma_Q_vx_phys**2, sigma_Q_vy_phys**2, sigma_ax_phys**2, sigma_ay_phys**2, sigma_bx_phys**2, sigma_by_phys**2])

#erreur_init = np.array([np.random.normal(0, sigma_x_init), np.random.normal(0, sigma_x_init), np.random.normal(0, sigma_v_init), np.random.normal(0, sigma_v_init), np.random.normal(0, sigma_a_init), np.random.normal(0, sigma_a_init), np.random.normal(0, sigma_b_init), np.random.normal(0, sigma_b_init), np.random.normal(0, sigma_x_init), np.random.normal(0, sigma_x_init), np.random.normal(0, sigma_v_init), np.random.normal(0, sigma_v_init), np.random.normal(0, sigma_a_init), np.random.normal(0, sigma_a_init), np.random.normal(0, sigma_b_init), np.random.normal(0, sigma_b_init), np.random.normal(0, sigma_x_init), np.random.normal(0, sigma_x_init), np.random.normal(0, sigma_v_init), np.random.normal(0, sigma_v_init), np.random.normal(0, sigma_a_init), np.random.normal(0, sigma_a_init), np.random.normal(0, sigma_b_init), np.random.normal(0, sigma_b_init)])

erreur_init_triche = np.zeros(24)
erreur_init_triche[0:2] = np.random.normal(0, sigma_init_large/10, size=2)
erreur_init_triche[8:10] = np.random.normal(0, sigma_init_large/10, size=2)
erreur_init_triche[16:18] = np.random.normal(0, sigma_init_large/10, size=2)

X_est_global = X_vrai.copy() + erreur_init_triche

drone1 = DistributedDrone(1, X_est_global[0:8], P_init_local, Q_local)
drone2 = DistributedDrone(2, X_est_global[8:16], P_init_local, Q_local)
drone3 = DistributedDrone(3, X_est_global[16:24], P_init_local, Q_local)

hist_vrai = np.zeros((n_steps, 24))
hist_vrai[0] = X_vrai
hist_est = np.zeros((n_steps, 24))
hist_est[0] = X_est_global
hist_capteur = np.zeros((n_steps_capteur, 7))
hist_vrai_capteur = np.zeros((n_steps_capteur, 24))
P_historique_1 = [drone1.P_est]
P_historique_2 = [drone2.P_est]
P_historique_3 = [drone3.P_est]
temps = []
temps_capteur =[]
step_capteur = 0

phi_x, phi_y = 0.0, 0.0
Ax1, Ay1, Ax2, Ay2, Ax3, Ay3 = 0.0, 2.0, 0.0, 2.0, 0.0, 2.0

# =====================================================================
# 4. BOUCLE DE SIMULATION TEMPORELLE
# =====================================================================
for step in range(1,n_steps):
    t = step * dt
    temps.append(t)
    
    # PHASE 1 : Le "Zig-Zag" d'Initialisation (0 à 4 secondes)
    # Objectif : Casser l'inobservabilité initiale et résoudre le "Triangle sur Pivot"
    # Action : Les drones accélèrent latéralement dans des directions différentes.
    if step < ((t_max/dt) / 4):
        # On donne une impulsion asymétrique
        Ax1, Ay1 = 1.0,  0.5
        Ax2, Ay2 = 1.0, -0.5  # Le drone 2 s'écarte vers le bas
        Ax3, Ay3 = 1.0,  0.8  # Le drone 3 s'écarte vers le haut
        
        u_vrai = np.array([Ax1, Ay1, Ax2, Ay2, Ax3, Ay3])
        u_kalman = u_vrai.copy()

    # PHASE 2 : La Respiration Déphasée (4 à 12 secondes)
    # Objectif : Maintenir l'observabilité maximale pendant les manœuvres
    # Action : Les drones tournent, mais avec des fréquences légèrement différentes
    elif step >= ((t_max/dt) / 4) and step < (3 * (t_max/dt) / 4):
        # Fréquences déphasées pour déformer continuellement le triangle
        omega_1 = 2.0  # Le Maître tourne doucement
        omega_2 = 2.5  # Le Drone 2 tourne un peu plus vite
        omega_3 = 1.5  # Le Drone 3 tourne plus lentement

        phi_x1 = omega_1 * t
        phi_y1 = omega_1 * t
        phi_x2 = omega_2 * t
        phi_y2 = omega_2 * t
        phi_x3 = omega_3 * t
        phi_y3 = omega_3 * t

        u_vrai = np.array([
            2.0 * np.cos(phi_x1), 2.0 * np.sin(phi_y1),
            2.0 * np.cos(phi_x2), 2.0 * np.sin(phi_y2),
            2.0 * np.cos(phi_x3), 2.0 * np.sin(phi_y3)
        ])
        u_kalman = u_vrai.copy()

    # PHASE 3 : Le Retour au Calme (12 à 16 secondes)
    # Objectif : Prouver que l'essaim a mémorisé sa géométrie
    # Action : Retour à une ligne droite lisse. L'incertitude va doucement remonter, mais sans retard.
    else:
        u_vrai = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        u_kalman = u_vrai.copy()

    u_d1 = u_vrai[0:2]; u_d2 = u_vrai[2:4]; u_d3 = u_vrai[4:6]

    w_vrai = np.zeros(24)

    w_vrai[[4, 12, 20]] = np.random.normal(0, sigma_ax_phys, size=3)
    w_vrai[[5, 13, 21]] = np.random.normal(0, sigma_ay_phys, size=3)
    X_vrai = F_vrai @ X_vrai + B_vrai @ u_vrai + w_vrai

    drone1.predict(u_d1, dt)
    drone2.predict(u_d2, dt)
    drone3.predict(u_d3, dt)

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

        dist_scalaire = [np.array([z_d12]), np.array([z_d23]), np.array([z_d13])]

        X_capteur = np.concatenate((z_gps_1, z_imu_2, dist_scalaire[0], dist_scalaire[1], dist_scalaire[2]), axis=None)

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

        hist_vrai_capteur[step_capteur] = X_vrai
        hist_capteur[step_capteur] = X_capteur 
        temps_capteur.append(t)
        step_capteur+=1
    # Sauvegarde des données synchronisées
    hist_vrai[step] = X_vrai
    hist_est[step] = np.concatenate((drone1.x_est, drone2.x_est, drone3.x_est))
    P_historique_1.append(drone1.P_est)
    P_historique_2.append(drone2.P_est)
    P_historique_3.append(drone3.P_est)

temps_np = np.array(temps)[:n_steps]
temps_capteur_np = np.array(temps_capteur)[:n_steps_capteur]
P_historique_2_np = np.array(P_historique_1)[:step]
P_historique_3_np = np.array(P_historique_2)[:step]
P_historique_1_np = np.array(P_historique_3)[:step]

x_vrai_1 = hist_vrai[:step, 0]
y_vrai_1 = hist_vrai[:step, 1]
vx_vrai_1 = hist_vrai[:step, 2]
vy_vrai_1 = hist_vrai[:step, 3]
ax_vrai_1 = hist_vrai[:step, 4]
ay_vrai_1 = hist_vrai[:step, 5]
bx_vrai_1 = hist_vrai[:step, 6]
by_vrai_1 = hist_vrai[:step, 7]

x_vrai_2 = hist_vrai[:step, 8]
y_vrai_2 = hist_vrai[:step, 9]
vx_vrai_2 = hist_vrai[:step, 10]
vy_vrai_2 = hist_vrai[:step, 11]
ax_vrai_2 = hist_vrai[:step, 12]
ay_vrai_2 = hist_vrai[:step, 13]
bx_vrai_2 = hist_vrai[:step, 14]
by_vrai_2 = hist_vrai[:step, 15]

x_vrai_3 = hist_vrai[:step, 16]
y_vrai_3 = hist_vrai[:step, 17]
vx_vrai_3 = hist_vrai[:step, 18]
vy_vrai_3 = hist_vrai[:step, 19]
ax_vrai_3 = hist_vrai[:step, 20]
ay_vrai_3 = hist_vrai[:step, 21]
bx_vrai_3 = hist_vrai[:step, 22]
by_vrai_3 = hist_vrai[:step, 23]

x_capteur_1 = hist_capteur[:step_capteur, 0]
y_capteur_1 = hist_capteur[:step_capteur, 1]

ax_capteur_2 = hist_capteur[:step_capteur, 2]
ay_capteur_2 = hist_capteur[:step_capteur, 3]

d12_capteur_2 = hist_capteur[:step_capteur, 4]
d23_capteur_2 = hist_capteur[:step_capteur, 5]

x_kalman_1 = hist_est[:step, 0]
y_kalman_1 = hist_est[:step, 1]
vx_kalman_1 = hist_est[:step, 2]
vy_kalman_1 = hist_est[:step, 3]
ax_kalman_1 = hist_est[:step, 4]
ay_kalman_1 = hist_est[:step, 5]
bx_kalman_1 = hist_est[:step, 6]
by_kalman_1 = hist_est[:step, 7]

x_kalman_2 = hist_est[:step, 8]
y_kalman_2 = hist_est[:step, 9]
vx_kalman_2 = hist_est[:step, 10]
vy_kalman_2 = hist_est[:step, 11]
ax_kalman_2 = hist_est[:step, 12]
ay_kalman_2 = hist_est[:step, 13]
bx_kalman_2 = hist_est[:step, 14]
by_kalman_2 = hist_est[:step, 15]

x_kalman_3 = hist_est[:step, 16]
y_kalman_3 = hist_est[:step, 17]
vx_kalman_3 = hist_est[:step, 18]
vy_kalman_3 = hist_est[:step, 19]
ax_kalman_3 = hist_est[:step, 20]
ay_kalman_3 = hist_est[:step, 21]
bx_kalman_3 = hist_est[:step, 22]
by_kalman_3 = hist_est[:step, 23]

x_vrai_capteur_x_1 = hist_vrai_capteur[:step_capteur, 0]
x_vrai_capteur_y_1 = hist_vrai_capteur[:step_capteur, 1]
x_vrai_capteur_ax_2 = hist_vrai_capteur[:step_capteur, 12]
x_vrai_capteur_ay_2 = hist_vrai_capteur[:step_capteur, 13]


x_vrai_pos_1 = hist_vrai[:step, 0:2]
x_kalman_pos_1 = hist_est[:step, 0:2]

x_vrai_pos_2 = hist_vrai[:step, 8:10]
x_kalman_pos_2 = hist_est[:step, 8:10]

x_vrai_pos_3 = hist_vrai[:step, 16:18]
x_kalman_pos_3 = hist_est[:step, 16:18]

mse_D1 = np.square(np.subtract(x_vrai_pos_1,x_kalman_pos_1)).mean()
mse_D1_init = np.square(np.subtract(x_vrai_pos_1[0],x_kalman_pos_1[0])).mean()
mse_D1_final = np.square(np.subtract(x_vrai_pos_1[step-1],x_kalman_pos_1[step-1])).mean()
mse_D2 = np.square(np.subtract(x_vrai_pos_2,x_kalman_pos_2)).mean()
mse_D2_init = np.square(np.subtract(x_vrai_pos_2[0],x_kalman_pos_2[0])).mean()
mse_D2_final = np.square(np.subtract(x_vrai_pos_2[step-1],x_kalman_pos_2[step-1])).mean()
mse_D3 = np.square(np.subtract(x_vrai_pos_3,x_kalman_pos_3)).mean()
mse_D3_init = np.square(np.subtract(x_vrai_pos_3[0],x_kalman_pos_3[0])).mean()
mse_D3_final = np.square(np.subtract(x_vrai_pos_3[step-1],x_kalman_pos_3[step-1])).mean()


print(f"MSE D1 : {mse_D1}")
print(f"MSE D1 initiale : {mse_D1_init}")
print(f"MSE D1 finale : {mse_D1_final}")
print(f"MSE D2 : {mse_D2}")
print(f"MSE D2 initiale : {mse_D2_init}")
print(f"MSE D2 finale : {mse_D2_final}")
print(f"MSE D3 : {mse_D3}")
print(f"MSE D3 initiale : {mse_D3_init}")
print(f"MSE D3 finale : {mse_D3_final}")
plot_drones = True
if plot_drones :

    fig, axs = plt.subplots(4, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Analyse EKF - Drone 1", fontsize=14, fontweight='bold')
    axs = axs.flatten()

    sigma = np.sqrt(P_historique_1_np[:, 0, 0])
    axs[0].plot(temps_np, x_kalman_1 - x_vrai_1, color='green', label='Estimation EKF')
    axs[0].scatter(temps_capteur_np, x_capteur_1 - x_vrai_capteur_x_1, color="red", marker="x", label="Mesures Drone 1", linewidths=0.5)
    axs[0].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[0].set_title("x_vrai - x_pred", fontsize=10)
    axs[0].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_1_np[:, 1, 1])
    axs[1].plot(temps_np, y_kalman_1 - y_vrai_1, color='green', label='Estimation EKF')
    axs[1].scatter(temps_capteur_np, y_capteur_1 - x_vrai_capteur_y_1, color="red", marker="x", label="Mesures Drone 1", linewidths=0.5)
    axs[1].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[1].set_title("y_vrai - y_pred", fontsize=10)
    axs[1].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_1_np[:, 2, 2])
    axs[2].plot(temps_np, vx_kalman_1 - vx_vrai_1, color='green', label='Estimation EKF')
    axs[2].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[2].set_title("vx_vrai - vy_pred", fontsize=10)
    axs[2].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_1_np[:, 3, 3])
    axs[3].plot(temps_np, vy_kalman_1 - vy_vrai_1, color='green', label='Estimation EKF')
    axs[3].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[3].set_title("vy_vrai - vy_pred", fontsize=10)
    axs[3].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_1_np[:, 4, 4])
    axs[4].plot(temps_np, ax_kalman_1 - ax_vrai_1, color='green', label='Estimation EKF')
    axs[4].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[4].set_title("ax_vrai - ax_pred", fontsize=10)
    axs[4].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_1_np[:, 5, 5])
    axs[5].plot(temps_np, ay_kalman_1 - ay_vrai_1, color='green', label='Estimation EKF')
    axs[5].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[5].set_title("ay_vrai - ay_pred", fontsize=10)
    axs[5].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_1_np[:, 6, 6])
    axs[6].plot(temps_np, bx_kalman_1 - bx_vrai_1, color='green', label='Estimation EKF')
    axs[6].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[6].set_title("bx_vrai - bx_pred", fontsize=10)
    axs[6].grid(True, linestyle=':', alpha=0.7)
    axs[6].set_xlabel("Temps (s)")

    sigma = np.sqrt(P_historique_1_np[:, 7, 7])
    axs[7].plot(temps_np, bx_kalman_1 - bx_vrai_1, color='green', label='Estimation EKF')
    axs[7].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[7].set_title("by_vrai - by_pred", fontsize=10)
    axs[7].grid(True, linestyle=':', alpha=0.7)
    axs[7].set_xlabel("Temps (s)")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.95))

    #############################################################################################

    fig, axs = plt.subplots(4, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Analyse EKF - Drone 2", fontsize=14, fontweight='bold')
    axs = axs.flatten()

    sigma = np.sqrt(P_historique_2_np[:, 0, 0])
    axs[0].plot(temps_np, x_kalman_2 - x_vrai_2, color='green', label='Estimation EKF')
    axs[0].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[0].set_title("x_vrai - x_pred", fontsize=10)
    axs[0].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_2_np[:, 1, 1])
    axs[1].plot(temps_np, y_kalman_2 - y_vrai_2, color='green', label='Estimation EKF')
    axs[1].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[1].set_title("y_vrai - y_pred", fontsize=10)
    axs[1].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_2_np[:, 2, 2])
    axs[2].plot(temps_np, vx_kalman_2 - vx_vrai_2, color='green', label='Estimation EKF')
    axs[2].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[2].set_title("vx_vrai - vx_pred", fontsize=10)
    axs[2].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_2_np[:, 3, 3])
    axs[3].plot(temps_np, vy_kalman_2 - vy_vrai_2, color='green', label='Estimation EKF')
    axs[3].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[3].set_title("vy_vrai - vy_pred", fontsize=10)
    axs[3].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_2_np[:, 4, 4])
    axs[4].plot(temps_np, ax_kalman_2 - ax_vrai_2, color='green', label='Estimation EKF')
    axs[4].scatter(temps_capteur_np, ax_capteur_2 - x_vrai_capteur_ax_2, color="red", marker="x", label='Mesures Drone 2', linewidths=0.5)
    axs[4].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[4].set_title("ax_vrai - ax_pred", fontsize=10)
    axs[4].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_2_np[:, 5, 5])
    axs[5].plot(temps_np, ay_kalman_2 - ay_vrai_2, color='green', label='Estimation EKF')
    axs[5].scatter(temps_capteur_np, ay_capteur_2 - x_vrai_capteur_ay_2, color="red", marker="x", label='Mesures Drone 2', linewidths=0.5)
    axs[5].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[5].set_title("ay_vrai - ay_pred", fontsize=10)
    axs[5].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_2_np[:, 6, 6])
    axs[6].plot(temps_np, bx_kalman_2 - bx_vrai_2, color='green', label='Estimation EKF')
    axs[6].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[6].set_title("bx_vrai - bx_pred", fontsize=10)
    axs[6].grid(True, linestyle=':', alpha=0.7)
    axs[6].set_xlabel("Temps (s)")

    sigma = np.sqrt(P_historique_2_np[:, 7, 7])
    axs[7].plot(temps_np, bx_kalman_2 - bx_vrai_2, color='green', label='Estimation EKF')
    axs[7].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[7].set_title("by_vrai - by_pred", fontsize=10)
    axs[7].grid(True, linestyle=':', alpha=0.7)
    axs[7].set_xlabel("Temps (s)")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.95))

    ####################################################################################################################

    fig, axs = plt.subplots(4, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Analyse EKF - Drone 3", fontsize=14, fontweight='bold')
    axs = axs.flatten()

    sigma = np.sqrt(P_historique_3_np[:, 0, 0])
    axs[0].plot(temps_np, x_kalman_3 - x_vrai_3, color='green', label='Estimation EKF')
    axs[0].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[0].set_title("x_vrai - x_pred", fontsize=10)
    axs[0].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_3_np[:, 1, 1])
    axs[1].plot(temps_np, y_kalman_3 - y_vrai_3, color='green', label='Estimation EKF')
    axs[1].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[1].set_title("y_vrai - y_pred", fontsize=10)
    axs[1].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_3_np[:, 2, 2])
    axs[2].plot(temps_np, vx_kalman_3 - vx_vrai_3, color='green', label='Estimation EKF')
    axs[2].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[2].set_title("vx_vrai - vx_pred", fontsize=10)
    axs[2].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_3_np[:, 3, 3])
    axs[3].plot(temps_np, vy_kalman_3 - vy_vrai_3, color='green', label='Estimation EKF')
    axs[3].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[3].set_title("vy_vrai - vy_pred", fontsize=10)
    axs[3].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_3_np[:, 4, 4])
    axs[4].plot(temps_np, ax_kalman_3 - ax_vrai_3, color='green', label='Estimation EKF')
    axs[4].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[4].set_title("ax_vrai - ax_pred", fontsize=10)
    axs[4].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_3_np[:, 5, 5])
    axs[5].plot(temps_np, ay_kalman_3 - ay_vrai_3, color='green', label='Estimation EKF')
    axs[5].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[5].set_title("ay_vrai - ay_pred", fontsize=10)
    axs[5].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_historique_3_np[:, 6, 6])
    axs[6].plot(temps_np, bx_kalman_3 - bx_vrai_3, color='green', label='Estimation EKF')
    axs[6].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[6].set_title("bx_vrai - bx_pred", fontsize=10)
    axs[6].grid(True, linestyle=':', alpha=0.7)
    axs[6].set_xlabel("Temps (s)")

    sigma = np.sqrt(P_historique_3_np[:, 7, 7])
    axs[7].plot(temps_np, bx_kalman_3 - bx_vrai_3, color='green', label='Estimation EKF')
    axs[7].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[7].set_title("by_vrai - by_pred", fontsize=10)
    axs[7].grid(True, linestyle=':', alpha=0.7)
    axs[7].set_xlabel("Temps (s)")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.95))

###################################################################################################################

plot_traj = True
if plot_traj : 
    plt.figure(figsize=(8,6))

    indice_inf = int((t_max/dt)/3)
    indice_sup = int(indice_inf*2)
    plt.scatter(x_capteur_1, y_capteur_1, color="red", marker="x", label="Mesures Drone 1", linewidths=0.5)
    plt.plot(x_vrai_1, y_vrai_1, marker='^', markevery=[indice_inf], label='Drone 1 vrai', color='black')
    plt.plot(x_vrai_2, y_vrai_2, marker='o', markevery=[indice_inf], label='Drone 2 vrai', color='black')
    plt.plot(x_vrai_3, y_vrai_3, marker='s', markevery=[indice_inf], label='Drone 3 vrai', color='black')
    plt.plot(x_kalman_1, y_kalman_1, marker='^', markevery=[indice_sup], label='Drone 1 corrige par Kalman', color='green', linestyle='-')
    plt.plot(x_kalman_2, y_kalman_2, marker='o', markevery=[indice_sup], label='Drone 2 corrige par Kalman', color='green', linestyle='-')
    plt.plot(x_kalman_3, y_kalman_3, marker='s', markevery=[indice_sup], label='Drone 3 corrige par Kalman', color='green', linestyle='-')

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.title("Trajectoire des 3 drones")
    plt.legend()
    plt.grid(True)

plt.show()
