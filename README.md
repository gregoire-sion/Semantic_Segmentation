# Semantic_Segmentation

import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres globaux ---
dt_imu = 0.01  
dt_gps = 0.1   
sim_time = 20.0
N_steps = int(sim_time / dt_imu)
time = np.arange(0, sim_time, dt_imu)

# Bruits standard pour tous les drones
Q_mat = np.diag([0.01, 0.01, 0.1, 0.1, np.deg2rad(1.0)]) ** 2
R_mat = np.diag([3.0, 3.0]) ** 2

# ==========================================
# CLASSE DRONE (Encapsulation de l'EKF)
# ==========================================
class DroneEKF:
    def __init__(self, drone_id, start_x, start_y, Q, R):
        self.id = drone_id
        
        # Matrices de bruit
        self.Q = Q
        self.R = R
        
        # Vérité terrain (Start Positions distinctes)
        self.x_true = np.array([[start_x], [start_y], [0.0], [0.0], [0.0]])
        
        # État estimé (initialisation identique à la vérité pour simplifier)
        self.x_est = np.array([[start_x], [start_y], [0.0], [0.0], [0.0]])
        self.P_est = np.eye(5) * 5.0 # Incertitude initiale
        
        # Historique pour l'affichage
        self.hx_true = []
        self.hx_est = []
        self.hz_gps = []
        self.hP_est = [] # Sauvegarde de P_est diagonal
        self.t_gps = []

    def simulate_truth_and_get_imu(self, true_ax, true_ay, true_omega, dt):
        """Mise à jour VÉRITÉ TERRAIN et génération IMU bruitée (Spécifique à CE drone)"""
        theta_true = self.x_true[4, 0]
        # Modèle physique parfait
        self.x_true[0, 0] += self.x_true[2, 0] * dt
        self.x_true[1, 0] += self.x_true[3, 0] * dt
        self.x_true[2, 0] += (true_ax * np.cos(theta_true) - true_ay * np.sin(theta_true)) * dt
        self.x_true[3, 0] += (true_ax * np.sin(theta_true) + true_ay * np.cos(theta_true)) * dt
        self.x_true[4, 0] += true_omega * dt
        
        # Génération des mesures IMU (bruitées, indépendantes)
        u_imu = np.array([[true_ax], [true_ay], [true_omega]]) + \
                np.random.multivariate_normal([0, 0, 0], np.diag([0.2, 0.2, 0.05])**2).reshape(3, 1)
        return u_imu

    def predict(self, u_imu, dt):
        """Étape 1 du Kalman : Prédiction avec l'IMU"""
        theta = self.x_est[4, 0]
        ax, ay, omega = u_imu[0, 0], u_imu[1, 0], u_imu[2, 0]
        
        # Modèle f(x, u)
        x_pred = np.zeros((5, 1))
        x_pred[0, 0] = self.x_est[0, 0] + self.x_est[2, 0] * dt
        x_pred[1, 0] = self.x_est[1, 0] + self.x_est[3, 0] * dt
        x_pred[2, 0] = self.x_est[2, 0] + (ax * np.cos(theta) - ay * np.sin(theta)) * dt
        x_pred[3, 0] = self.x_est[3, 0] + (ax * np.sin(theta) + ay * np.cos(theta)) * dt
        x_pred[4, 0] = self.x_est[4, 0] + omega * dt
        
        # Jacobienne F
        F = np.eye(5)
        F[0, 2] = dt
        F[1, 3] = dt
        F[2, 4] = (-ax * np.sin(theta) - ay * np.cos(theta)) * dt
        F[3, 4] = ( ax * np.cos(theta) - ay * np.sin(theta)) * dt
        
        # P_pred = F P_est F.T + Q
        self.P_est = F @ self.P_est @ F.T + self.Q
        self.x_est = x_pred

    def update_gps(self, t):
        """Étape 2 du Kalman : Mise à jour avec le GPS (Tirage indépendant)"""
        # Simulation d'une mesure GPS avec du bruit (Spécifique à CE drone)
        z_gps = np.array([[self.x_true[0, 0]], [self.x_true[1, 0]]]) + \
                np.random.multivariate_normal([0, 0], self.R).reshape(2, 1)
        
        self.hz_gps.append([z_gps[0, 0], z_gps[1, 0]])
        self.t_gps.append(t)
        
        # Modèle d'observation linéaire
        H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0]
        ])
        
        # Innovation S K
        y = z_gps - (H @ self.x_est)
        S = H @ self.P_est @ H.T + self.R
        K = self.P_est @ H.T @ np.linalg.inv(S)
        
        # Correction de l'état
        self.x_est = self.x_est + K @ y
        self.P_est = (np.eye(5) - K @ H) @ self.P_est

    def save_history(self):
        """Stockage des états et de la covariance pour l'affichage"""
        self.hx_true.append(self.x_true.copy())
        self.hx_est.append(self.x_est.copy())
        self.hP_est.append(np.diag(self.P_est).copy()) # Covariance diagonale

# ==========================================
# INITIALISATION DES DRONES
# ==========================================
# Deux drones avec un décalage en Y de 10 mètres
drone1 = DroneEKF(drone_id=1, start_x=0.0, start_y=0.0, Q=Q_mat, R=R_mat)
drone2 = DroneEKF(drone_id=2, start_x=0.0, start_y=10.0, Q=Q_mat, R=R_mat)

drones = [drone1, drone2] # Liste pour faciliter les boucles

# ==========================================
# BOUCLE DE SIMULATION PRINCIPALE
# ==========================================
for i in range(N_steps):
    t_actuel = i * dt_imu
    
    # Consignes de vol (Trajectoire identique pour les deux drones)
    ax, ay, omega = 0.5, 0.0, 0.3
    
    # Mise à jour de CHAQUE drone indépendamment
    for drone in drones:
        # 0. Vérité + IMU bruitée spécifique
        u_imu = drone.simulate_truth_and_get_imu(ax, ay, omega, dt_imu)
        
        # 1. Prédiction EKF
        drone.predict(u_imu, dt_imu)
        
        # 2. Mise à jour GPS EKF (10Hz, indépendant)
        if i % 10 == 0:
            drone.update_gps(t_actuel)
            
        # 3. Stockage
        drone.save_history()

# ==========================================
# FORMATAGE DES DONNÉES POUR L'AFFICHAGE
# ==========================================
d1_true = np.array(drone1.hx_true).squeeze()
d1_est = np.array(drone1.hx_est).squeeze()
d1_gps = np.array(drone1.hz_gps)
d1_P = np.array(drone1.hP_est)

d2_true = np.array(drone2.hx_true).squeeze()
d2_est = np.array(drone2.hx_est).squeeze()
d2_gps = np.array(drone2.hz_gps)
d2_P = np.array(drone2.hP_est)

# ==========================================
# AFFICHAGE DU STYLE COMPLET
# ==========================================
fig = plt.figure(figsize=(18, 14))

# --- Plot 1 : Trajectoires 2D indépendantes ---
plt.subplot(3, 2, 1)
# Drone 1 (Bleu)
plt.plot(d1_true[:,0], d1_true[:, 1], 'k--', linewidth=1, label="D1 Réel")
plt.plot(d1_est[:,0], d1_est[:, 1], 'b-', linewidth=2, label="D1 Estimé (EKF)")
plt.scatter(d1_gps[:, 0], d1_gps[:, 1], color='blue', marker='x', s=10, alpha=0.2, label="D1 GPS")

# Drone 2 (Rouge)
plt.plot(d2_true[:,0], d2_true[:, 1], 'k--', linewidth=1, label="D2 Réel")
plt.plot(d2_est[:,0], d2_est[:, 1], 'r-', linewidth=2, label="D2 Estimé (EKF)")
plt.scatter(d2_gps[:, 0], d2_gps[:, 1], color='red', marker='x', s=10, alpha=0.2, label="D2 GPS")

plt.title("Trajectoire 2D: Drone 1 vs Drone 2 (Indépendants)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.grid(True)
plt.axis('equal')

labels = ["Position X (m)", "Position Y (m)", "Vitesse Vx (m/s)", "Vitesse Vy (m/s)", "Cap Theta (rad)"]

# ==========================================
# Nouveaux Plots: 5 Variables d'état
# ==========================================
for j in range(5):
    plt.subplot(3, 2, j + 2)
    
    # --- Drone 1 (Bleu) ---
    plt.plot(time, d1_true[:, j], 'k--', alpha=0.5) # Réel D1
    plt.plot(time, d1_est[:, j], 'b-', label="D1 Estimé") # Estimé D1
    # Couloir 3-sigma Drone 1
    sigma1 = np.sqrt(d1_P[:, j])
    plt.fill_between(time, d1_est[:, j] - 3*sigma1, d1_est[:, j] + 3*sigma1, color='blue', alpha=0.1)

    # --- Drone 2 (Rouge) ---
    plt.plot(time, d2_true[:, j], 'k--', alpha=0.5) # Réel D2
    plt.plot(time, d2_est[:, j], 'r-', label="D2 Estimé") # Estimé D2
    # Couloir 3-sigma Drone 2
    sigma2 = np.sqrt(d2_P[:, j])
    plt.fill_between(time, d2_est[:, j] - 3*sigma2, d2_est[:, j] + 3*sigma2, color='red', alpha=0.1)
    
    # Ajout mesures GPS brutes pour X et Y
    if j < 2:
        plt.scatter(drone1.t_gps, d1_gps[:, j], color='blue', marker='x', s=10, alpha=0.3)
        plt.scatter(drone2.t_gps, d2_gps[:, j], color='red', marker='x', s=10, alpha=0.3)
        
    plt.title(f"{labels[j]} - D1 vs D2")
    plt.xlabel("Temps (s)")
    plt.grid(True)
    if j == 0:
        plt.legend(loc="upper left")

plt.tight_layout()
plt.show()
