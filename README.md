import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PARAMÈTRES GLOBAUX
# ==========================================
dt_imu = 0.01   # 100 Hz
dt_gps = 2.0    # GPS lent (0.5 Hz) pour créer de gros sauts
dt_uwb = 0.1    # Capteur de distance rapide (10 Hz)
sim_time = 30.0 

# 🔴 LE SWITCH 🔴
USE_RANGING = True  

step_gps = int(dt_gps / dt_imu)
step_uwb = int(dt_uwb / dt_imu)
N_steps = int(sim_time / dt_imu)
time = np.arange(0, sim_time, dt_imu)

# ==========================================
# CLASSE SWARM EKF (Baseline Simple 10x10)
# ==========================================
class SwarmEKF_Baseline:
    def __init__(self, start_y2=10.0):
        # Bruits standard
        q = np.diag([0.01, 0.01, 0.1, 0.1, np.deg2rad(1.0)]) ** 2
        self.Q = np.block([[q, np.zeros((5,5))], [np.zeros((5,5)), q]])
        self.R_gps = np.diag([3.0, 3.0, 3.0, 3.0]) ** 2
        self.R_dist = np.array([[0.5 ** 2]]) # Incertitude UWB
        
        # Initialisation Vérité et Estimation (10 variables)
        self.X_true = np.zeros((10, 1))
        self.X_true[6, 0] = start_y2 
        
        self.X_est = np.zeros((10, 1))
        self.X_est[6, 0] = start_y2
        self.P_est = np.eye(10) * 5.0
        
        # Historique
        self.hX_true, self.hX_est, self.hP_est = [], [], []

    def simulate_truth_and_get_imu(self, dt):
        ax, ay, omega = 0.5, 0.0, 0.3
        
        # Mise à jour de la vérité pour D1 (index 0) et D2 (index 5)
        for i in [0, 5]: 
            theta = self.X_true[i+4, 0]
            self.X_true[i, 0]   += self.X_true[i+2, 0] * dt
            self.X_true[i+1, 0] += self.X_true[i+3, 0] * dt
            self.X_true[i+2, 0] += (ax * np.cos(theta) - ay * np.sin(theta)) * dt
            self.X_true[i+3, 0] += (ax * np.sin(theta) + ay * np.cos(theta)) * dt
            self.X_true[i+4, 0] += omega * dt
            
        # IMU D1 (Parfaite + Bruit blanc)
        u1 = np.array([[ax], [ay], [omega]]) + np.random.normal(0, 0.1, (3,1))
        
        # IMU D2 (Défectueuse : Biais + Bruit blanc)
        bias_d2 = np.array([[0.1], [0.05], [np.deg2rad(0.5)]])
        u2 = np.array([[ax], [ay], [omega]]) + bias_d2 + np.random.normal(0, 0.1, (3,1))
        
        return u1, u2

    def predict(self, u1, u2, dt):
        X_pred = np.zeros((10, 1))
        F = np.eye(10)
        
        for idx, u in zip([0, 5], [u1, u2]):
            x, y, vx, vy, theta = self.X_est[idx:idx+5, 0]
            a_x, a_y, om = u[0,0], u[1,0], u[2,0]
            
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

    def update_gps(self):
        # Mesure brute X, Y pour les deux drones
        Z = np.vstack((self.X_true[0:2], self.X_true[5:7])) + np.random.normal(0, 3, (4,1))
        
        H = np.zeros((4, 10))
        H[0, 0] = 1; H[1, 1] = 1
        H[2, 5] = 1; H[3, 6] = 1
        
        y = Z - (H @ self.X_est)
        S = H @ self.P_est @ H.T + self.R_gps
        K = self.P_est @ H.T @ np.linalg.inv(S)
        
        self.X_est = self.X_est + K @ y
        self.P_est = (np.eye(10) - K @ H) @ self.P_est

    def update_distance(self):
        # Mesure brute de la distance relative
        true_dist = np.sqrt((self.X_true[5,0]-self.X_true[0,0])**2 + (self.X_true[6,0]-self.X_true[1,0])**2)
        z_dist = true_dist + np.random.normal(0, np.sqrt(self.R_dist[0,0]))
        
        e_dist = np.sqrt((self.X_est[5,0]-self.X_est[0,0])**2 + (self.X_est[6,0]-self.X_est[1,0])**2)
        
        H = np.zeros((1, 10))
        if e_dist > 0.01:
            H[0, 0] = -(self.X_est[5,0]-self.X_est[0,0]) / e_dist
            H[0, 1] = -(self.X_est[6,0]-self.X_est[1,0]) / e_dist
            H[0, 5] =  (self.X_est[5,0]-self.X_est[0,0]) / e_dist
            H[0, 6] =  (self.X_est[6,0]-self.X_est[1,0]) / e_dist
            
        y = z_dist - e_dist
        S = H @ self.P_est @ H.T + self.R_dist
        K = self.P_est @ H.T @ np.linalg.inv(S)
        
        self.X_est = self.X_est + K * y
        self.P_est = (np.eye(10) - K @ H) @ self.P_est

    def save_history(self):
        self.hX_true.append(self.X_true.copy())
        self.hX_est.append(self.X_est.copy())
        self.hP_est.append(np.diag(self.P_est).copy())

# ==========================================
# BOUCLE PRINCIPALE
# ==========================================
swarm = SwarmEKF_Baseline()

for i in range(N_steps):
    u1, u2 = swarm.simulate_truth_and_get_imu(dt_imu)
    swarm.predict(u1, u2, dt_imu)
    
    if i % step_gps == 0:
        swarm.update_gps()
        
    if USE_RANGING and i % step_uwb == 0:
        swarm.update_distance()
        
    swarm.save_history()

# ==========================================
# FORMATAGE DES DONNÉES
# ==========================================
X_t = np.array(swarm.hX_true).squeeze()
X_e = np.array(swarm.hX_est).squeeze()
P_e = np.array(swarm.hP_est)

dist_t = np.sqrt((X_t[:,5]-X_t[:,0])**2 + (X_t[:,6]-X_t[:,1])**2)
dist_e = np.sqrt((X_e[:,5]-X_e[:,0])**2 + (X_e[:,6]-X_e[:,1])**2)

# ==========================================
# AFFICHAGE PROPRE (Bug de la vérité corrigé)
# ==========================================
fig = plt.figure(figsize=(16, 18))
plt.subplots_adjust(hspace=0.4)

état_txt = "AVEC UWB (Couplé)" if USE_RANGING else "SANS UWB (Dérive totale)"

# --- 1. Trajectoire 2D ---
plt.subplot(4, 2, 1)
plt.plot(X_t[:,0], X_t[:,1], 'k--', label="D1 Réel")
plt.plot(X_e[:,0], X_e[:,1], 'b-', label="D1 Est")
plt.plot(X_t[:,5], X_t[:,6], 'k:', linewidth=2, label="D2 Réel") # Changé en pointillés
plt.plot(X_e[:,5], X_e[:,6], 'r-', label="D2 Est (Biaisé)")
plt.title(f"Trajectoire 2D ({état_txt})"); plt.xlabel("X (m)"); plt.ylabel("Y (m)")
plt.legend(); plt.grid(True); plt.axis('equal')

# --- 2. Distance ---
plt.subplot(4, 2, 2)
plt.plot(time, dist_t, 'k--', label="Distance Réelle")
plt.plot(time, dist_e, 'g-', label="Distance Estimée")
plt.title(f"Distance D1-D2 ({état_txt})"); plt.xlabel("Temps (s)"); plt.ylabel("Distance (m)")
plt.legend(); plt.grid(True)

# --- 3 à 7. Les 5 variables d'états ---
labels = ["Position X (m)", "Position Y (m)", "Vitesse Vx (m/s)", "Vitesse Vy (m/s)", "Cap Theta (rad)"]

for j in range(5):
    plt.subplot(4, 2, j + 3)
    
    # Drone 1 (Vérité et Estimé)
    plt.plot(time, X_t[:, j], 'k--', alpha=0.5, label="D1 Réel" if j==0 else "")
    plt.plot(time, X_e[:, j], 'b-', label="D1 Est" if j==0 else "")
    s1 = np.sqrt(P_e[:, j])
    plt.fill_between(time, X_e[:, j]-3*s1, X_e[:, j]+3*s1, color='blue', alpha=0.1)
    
    # Drone 2 (Vérité et Estimé) -> C'était le bug ! C'est X_t[:, j+5]
    plt.plot(time, X_t[:, j+5], 'k:', alpha=0.7, label="D2 Réel" if j==0 else "")
    plt.plot(time, X_e[:, j+5], 'r-', label="D2 Est" if j==0 else "")
    s2 = np.sqrt(P_e[:, j+5])
    plt.fill_between(time, X_e[:, j+5]-3*s2, X_e[:, j+5]+3*s2, color='red', alpha=0.1)

    plt.title(labels[j]); plt.xlabel("Temps (s)"); plt.grid(True)
    if j == 0: plt.legend(loc="upper left")

plt.savefig(f"baseline_simple_ranging_{USE_RANGING}.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Terminé ! Image sauvegardée : baseline_simple_ranging_{USE_RANGING}.png")


