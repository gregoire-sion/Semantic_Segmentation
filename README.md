import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PARAMÈTRES GLOBAUX & SWITCH
# ==========================================
dt_imu = 0.01  
dt_gps = 0.1   
sim_time = 20.0

# 🔴 LE BOOLÉEN MAGIQUE 🔴
USE_RANGING = True  # Mets True pour coupler les drones, False pour les rendre indépendants

N_steps = int(sim_time / dt_imu)
time = np.arange(0, sim_time, dt_imu)

# ==========================================
# CLASSE SWARM EKF (Filtre Unifié 10x10)
# ==========================================
class SwarmEKF:
    def __init__(self, start_y2=10.0):
        # Bruits
        q = np.diag([0.01, 0.01, 0.1, 0.1, np.deg2rad(1.0)]) ** 2
        self.Q = np.block([[q, np.zeros((5,5))], 
                           [np.zeros((5,5)), q]]) # Matrice 10x10
        
        self.R_gps = np.diag([3.0, 3.0, 3.0, 3.0]) ** 2 
        self.R_dist = np.array([[0.5 ** 2]]) # Capteur UWB (ex: 50 cm d'incertitude)
        
        # Vérité terrain
        self.X_true = np.zeros((10, 1))
        self.X_true[6, 0] = start_y2 
        
        # État estimé 
        self.X_est = np.zeros((10, 1))
        self.X_est[6, 0] = start_y2
        self.P_est = np.eye(10) * 5.0 
        
        # Historique
        self.hX_true, self.hX_est, self.hP_est = [], [], []
        self.hz_gps1, self.hz_gps2, self.t_gps = [], [], []

    def simulate_truth_and_get_imu(self, dt):
        """Met à jour la vérité et génère les IMU des deux drones"""
        ax, ay, omega = 0.5, 0.0, 0.3
        
        for i in [0, 5]: 
            theta_true = self.X_true[i+4, 0]
            self.X_true[i, 0]   += self.X_true[i+2, 0] * dt
            self.X_true[i+1, 0] += self.X_true[i+3, 0] * dt
            self.X_true[i+2, 0] += (ax * np.cos(theta_true) - ay * np.sin(theta_true)) * dt
            self.X_true[i+3, 0] += (ax * np.sin(theta_true) + ay * np.cos(theta_true)) * dt
            self.X_true[i+4, 0] += omega * dt
            
        u_imu1 = np.array([[ax], [ay], [omega]]) + np.random.multivariate_normal([0,0,0], np.diag([0.2, 0.2, 0.05])**2).reshape(3,1)
        u_imu2 = np.array([[ax], [ay], [omega]]) + np.random.multivariate_normal([0,0,0], np.diag([0.2, 0.2, 0.05])**2).reshape(3,1)
        
        return u_imu1, u_imu2

    def predict(self, u_imu1, u_imu2, dt):
        """Prédiction combinée 10x10"""
        X_pred = np.zeros((10, 1))
        F = np.eye(10)
        
        for idx, u_imu in zip([0, 5], [u_imu1, u_imu2]):
            x, y, vx, vy, theta = self.X_est[idx:idx+5, 0]
            a_x, a_y, om = u_imu[0,0], u_imu[1,0], u_imu[2,0]
            
            X_pred[idx, 0]   = x + vx * dt
            X_pred[idx+1, 0] = y + vy * dt
            X_pred[idx+2, 0] = vx + (a_x * np.cos(theta) - a_y * np.sin(theta)) * dt
            X_pred[idx+3, 0] = vy + (a_x * np.sin(theta) + a_y * np.cos(theta)) * dt
            X_pred[idx+4, 0] = theta + om * dt
            
            F[idx,   idx+2] = dt
            F[idx+1, idx+3] = dt
            F[idx+2, idx+4] = (-a_x * np.sin(theta) - a_y * np.cos(theta)) * dt
            F[idx+3, idx+4] = ( a_x * np.cos(theta) - a_y * np.sin(theta)) * dt
            
        self.P_est = F @ self.P_est @ F.T + self.Q
        self.X_est = X_pred

    def update_gps(self, t):
        """Mise à jour GPS combinée"""
        z_gps1 = self.X_true[0:2] + np.random.multivariate_normal([0,0], np.diag([3.0, 3.0])**2).reshape(2,1)
        z_gps2 = self.X_true[5:7] + np.random.multivariate_normal([0,0], np.diag([3.0, 3.0])**2).reshape(2,1)
        
        self.hz_gps1.append([z_gps1[0,0], z_gps1[1,0]])
        self.hz_gps2.append([z_gps2[0,0], z_gps2[1,0]])
        self.t_gps.append(t)
        
        Z = np.vstack((z_gps1, z_gps2)) 
        
        H_gps = np.zeros((4, 10))
        H_gps[0, 0] = 1.0; H_gps[1, 1] = 1.0 
        H_gps[2, 5] = 1.0; H_gps[3, 6] = 1.0 
        
        y = Z - (H_gps @ self.X_est)
        S = H_gps @ self.P_est @ H_gps.T + self.R_gps
        K = self.P_est @ H_gps.T @ np.linalg.inv(S)
        
        self.X_est = self.X_est + K @ y
        self.P_est = (np.eye(10) - K @ H_gps) @ self.P_est

    def update_distance(self):
        """Mise à jour avec la mesure de distance relative"""
        x1, y1 = self.X_true[0, 0], self.X_true[1, 0]
        x2, y2 = self.X_true[5, 0], self.X_true[6, 0]
        
        true_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        z_dist = true_dist + np.random.normal(0, np.sqrt(self.R_dist[0,0]))
        
        e_x1, e_y1 = self.X_est[0, 0], self.X_est[1, 0]
        e_x2, e_y2 = self.X_est[5, 0], self.X_est[6, 0]
        est_dist = np.sqrt((e_x2 - e_x1)**2 + (e_y2 - e_y1)**2)
        
        H_dist = np.zeros((1, 10))
        if est_dist > 0.01: 
            H_dist[0, 0] = -(e_x2 - e_x1) / est_dist  
            H_dist[0, 1] = -(e_y2 - e_y1) / est_dist  
            H_dist[0, 5] =  (e_x2 - e_x1) / est_dist  
            H_dist[0, 6] =  (e_y2 - e_y1) / est_dist  
            
        y = z_dist - est_dist
        S = H_dist @ self.P_est @ H_dist.T + self.R_dist
        K = self.P_est @ H_dist.T @ np.linalg.inv(S)
        
        self.X_est = self.X_est + K * y
        self.P_est = (np.eye(10) - K @ H_dist) @ self.P_est

    def save_history(self):
        self.hX_true.append(self.X_true.copy())
        self.hX_est.append(self.X_est.copy())
        self.hP_est.append(np.diag(self.P_est).copy())

# ==========================================
# SIMULATION
# ==========================================
swarm = SwarmEKF(start_y2=10.0)

for i in range(N_steps):
    t_actuel = i * dt_imu
    
    u_imu1, u_imu2 = swarm.simulate_truth_and_get_imu(dt_imu)
    swarm.predict(u_imu1, u_imu2, dt_imu)
    
    if i % 10 == 0:
        swarm.update_gps(t_actuel)
        
        # 🔴 APPLICATION DU SWITCH ICI 🔴
        if USE_RANGING:
            swarm.update_distance() 
            
    swarm.save_history()

# ==========================================
# FORMATAGE DES DONNÉES
# ==========================================
X_t = np.array(swarm.hX_true).squeeze()
X_e = np.array(swarm.hX_est).squeeze()
P_e = np.array(swarm.hP_est)
z_g1 = np.array(swarm.hz_gps1)
z_g2 = np.array(swarm.hz_gps2)

dist_true = np.sqrt((X_t[:, 5] - X_t[:, 0])**2 + (X_t[:, 6] - X_t[:, 1])**2)
dist_est = np.sqrt((X_e[:, 5] - X_e[:, 0])**2 + (X_e[:, 6] - X_e[:, 1])**2)

# ==========================================
# AFFICHAGE 4x2
# ==========================================
fig = plt.figure(figsize=(18, 18))

# Titre dynamique selon l'état du booléen
titre_etat = "AVEC Couplage de Distance (Ranging ON)" if USE_RANGING else "SANS Couplage de Distance (Ranging OFF)"

# --- Plot 1 : Trajectoires 2D ---
plt.subplot(4, 2, 1)
plt.plot(X_t[:,0], X_t[:, 1], 'k--', linewidth=1, label="D1 Réel")
plt.plot(X_e[:,0], X_e[:, 1], 'b-', linewidth=2, label="D1 Estimé")
plt.scatter(z_g1[:, 0], z_g1[:, 1], color='blue', marker='x', s=10, alpha=0.2)

plt.plot(X_t[:,5], X_t[:, 6], 'k--', linewidth=1, label="D2 Réel")
plt.plot(X_e[:,5], X_e[:, 6], 'r-', linewidth=2, label="D2 Estimé")
plt.scatter(z_g2[:, 0], z_g2[:, 1], color='red', marker='x', s=10, alpha=0.2)

plt.title(f"Trajectoire 2D : {titre_etat}")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.grid(True)
plt.axis('equal')

# --- Plot 2 : Erreur de distance relative ---
plt.subplot(4, 2, 2)
plt.plot(time, dist_true, 'k--', linewidth=2, label="Distance Réelle (10m)")
plt.plot(time, dist_est, 'g-', linewidth=2, label="Distance Estimée")
plt.title(f"Distance relative D1 <-> D2 : {titre_etat}")
plt.xlabel("Temps (s)")
plt.ylabel("Distance (m)")
plt.legend()
plt.grid(True)

# --- 5 Variables d'état ---
labels = ["Position X (m)", "Position Y (m)", "Vitesse Vx (m/s)", "Vitesse Vy (m/s)", "Cap Theta (rad)"]

for j in range(5):
    plt.subplot(4, 2, j + 3) 
    
    # Drone 1 (Bleu)
    plt.plot(time, X_t[:, j], 'k--', alpha=0.5)
    plt.plot(time, X_e[:, j], 'b-', label="D1 Estimé")
    s1 = np.sqrt(P_e[:, j])
    plt.fill_between(time, X_e[:, j] - 3*s1, X_e[:, j] + 3*s1, color='blue', alpha=0.1)

    # Drone 2 (Rouge)
    plt.plot(time, X_t[:, j+5], 'k--', alpha=0.5)
    plt.plot(time, X_e[:, j+5], 'r-', label="D2 Estimé")
    s2 = np.sqrt(P_e[:, j+5])
    plt.fill_between(time, X_e[:, j+5] - 3*s2, X_e[:, j+5] + 3*s2, color='red', alpha=0.1)
    
    if j < 2:
        plt.scatter(swarm.t_gps, z_g1[:, j], color='blue', marker='x', s=10, alpha=0.3)
        plt.scatter(swarm.t_gps, z_g2[:, j], color='red', marker='x', s=10, alpha=0.3)
        
    plt.title(f"{labels[j]} - D1 vs D2")
    plt.xlabel("Temps (s)")
    plt.grid(True)
    if j == 0:
        plt.legend(loc="upper left")

plt.tight_layout()

# ==========================================
# SAUVEGARDE DE L'IMAGE 
# ==========================================
filename = f"resultat_ekf_ranging_{USE_RANGING}.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"✅ Simulation terminée ! Graphique sauvegardé sous le nom : {filename}")

plt.close(fig) 
