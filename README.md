import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PARAMÈTRES GLOBAUX (Totalement adaptables)
# ==========================================
dt_imu = 0.01   # 100 Hz
dt_gps = 2.0    # <--- CHANGE MOI : 0.1 pour rapide, 2.0 pour lent
dt_uwb = 0.1    # 10 Hz pour la distance relative

USE_RANGING = True 
sim_time = 30.0

# Calcul automatique des intervalles (pour que le code soit dynamique)
step_gps = int(dt_gps / dt_imu)
step_uwb = int(dt_uwb / dt_imu)

N_steps = int(sim_time / dt_imu)
time = np.arange(0, sim_time, dt_imu)

# ==========================================
# CLASSE SWARM EKF (Vecteur d'état 16 variables)
# [x, y, vx, vy, theta, b_ax, b_ay, b_om] x 2
# ==========================================
class SwarmEKFExpert:
    def __init__(self, start_y2=10.0):
        # Q : On ajoute une petite incertitude sur la stabilité des biais
        q_std = np.array([0.01, 0.01, 0.1, 0.1, np.deg2rad(1.0), 0.001, 0.001, 0.0001])
        q_block = np.diag(q_std**2)
        self.Q = np.block([[q_block, np.zeros((8,8))], [np.zeros((8,8)), q_block]])
        
        self.R_gps = np.diag([3.0, 3.0, 3.0, 3.0]) ** 2 
        self.R_dist = np.array([[0.5 ** 2]])
        
        # État : 16 variables
        self.X_true = np.zeros((16, 1))
        self.X_true[8, 0] = start_y2 # Drone 2 en Y=10
        # On définit les vrais biais (ce que le filtre doit découvrir)
        self.true_bias2 = np.array([0.1, 0.05, np.deg2rad(0.5)]) 
        
        self.X_est = np.zeros((16, 1))
        self.X_est[8, 0] = start_y2
        self.P_est = np.eye(16) * 5.0
        
        self.hX_true, self.hX_est, self.hP_est = [], [], []
        self.hz_gps1, self.hz_gps2, self.t_gps = [], [], []

    def simulate_truth_and_get_imu(self, dt):
        ax, ay, omega = 0.5, 0.0, 0.3
        for i in [0, 8]: # Update D1 et D2
            theta = self.X_true[i+4, 0]
            self.X_true[i:i+2, 0] += self.X_true[i+2:i+4, 0].flatten() * dt
            self.X_true[i+2, 0] += (ax * np.cos(theta) - ay * np.sin(theta)) * dt
            self.X_true[i+3, 0] += (ax * np.sin(theta) + ay * np.cos(theta)) * dt
            self.X_true[i+4, 0] += omega * dt
            
        u1 = np.array([[ax], [ay], [omega]]) + np.random.normal(0, 0.1, (3,1))
        # Drone 2 a un BIAIS réel
        u2 = np.array([[ax], [ay], [omega]]) + self.true_bias2.reshape(3,1) + np.random.normal(0, 0.1, (3,1))
        return u1, u2

    def predict(self, u1, u2, dt):
        X_pred = self.X_est.copy()
        F = np.eye(16)
        
        for idx, u in zip([0, 8], [u1, u2]):
            x, y, vx, vy, theta = self.X_est[idx:idx+5, 0].flatten()
            # On retire le biais estimé du signal IMU brut !
            b_ax, b_ay, b_om = self.X_est[idx+5:idx+8, 0].flatten()
            a_x, a_y, om = u[0,0] - b_ax, u[1,0] - b_ay, u[2,0] - b_om
            
            X_pred[idx, 0]   += vx * dt
            X_pred[idx+1, 0] += vy * dt
            X_pred[idx+2, 0] += (a_x * np.cos(theta) - a_y * np.sin(theta)) * dt
            X_pred[idx+3, 0] += (a_x * np.sin(theta) + a_y * np.cos(theta)) * dt
            X_pred[idx+4, 0] += om * dt
            
            F[idx, idx+2] = dt; F[idx+1, idx+3] = dt
            F[idx+2, idx+4] = (-a_x * np.sin(theta) - a_y * np.cos(theta)) * dt
            F[idx+3, idx+4] = ( a_x * np.cos(theta) - a_y * np.sin(theta)) * dt
            # Jacobienne par rapport aux biais
            F[idx+2, idx+5] = -np.cos(theta) * dt; F[idx+2, idx+6] = np.sin(theta) * dt
            F[idx+3, idx+5] = -np.sin(theta) * dt; F[idx+3, idx+6] = -np.cos(theta) * dt
            F[idx+4, idx+7] = -dt

        self.P_est = F @ self.P_est @ F.T + self.Q
        self.X_est = X_pred

    def update_gps(self, t):
        Z = np.vstack((self.X_true[0:2], self.X_true[8:10])) + np.random.normal(0, 3, (4,1))
        self.hz_gps1.append([Z[0,0], Z[1,0]]); self.hz_gps2.append([Z[2,0], Z[3,0]]); self.t_gps.append(t)
        H = np.zeros((4, 16))
        H[0,0]=1; H[1,1]=1; H[2,8]=1; H[3,9]=1
        K = self.P_est @ H.T @ np.linalg.inv(H @ self.P_est @ H.T + self.R_gps)
        self.X_est += K @ (Z - H @ self.X_est)
        self.P_est = (np.eye(16) - K @ H) @ self.P_est

    def update_distance(self):
        z_dist = np.sqrt((self.X_true[8,0]-self.X_true[0,0])**2 + (self.X_true[9,0]-self.X_true[1,0])**2) + np.random.normal(0, 0.5)
        e_dist = np.sqrt((self.X_est[8,0]-self.X_est[0,0])**2 + (self.X_est[9,0]-self.X_est[1,0])**2)
        H = np.zeros((1, 16))
        H[0,0] = -(self.X_est[8,0]-self.X_est[0,0])/e_dist; H[0,1] = -(self.X_est[9,0]-self.X_est[1,0])/e_dist
        H[0,8] = -H[0,0]; H[0,9] = -H[0,1]
        K = self.P_est @ H.T @ np.linalg.inv(H @ self.P_est @ H.T + self.R_dist)
        self.X_est += K * (z_dist - e_dist)
        self.P_est = (np.eye(16) - K @ H) @ self.P_est

    def save_history(self):
        self.hX_true.append(self.X_true.copy()); self.hX_est.append(self.X_est.copy())
        self.hP_est.append(np.diag(self.P_est).copy())

# --- BOUCLE ---
swarm = SwarmEKFExpert()
for i in range(N_steps):
    u1, u2 = swarm.simulate_truth_and_get_imu(dt_imu)
    swarm.predict(u1, u2, dt_imu)
    if i % step_gps == 0: swarm.update_gps(i*dt_imu)
    if USE_RANGING and i % step_uwb == 0: swarm.update_distance()
    swarm.save_history()

# --- PLOTS ---
X_t, X_e, P_e = np.array(swarm.hX_true).squeeze(), np.array(swarm.hX_est).squeeze(), np.array(swarm.hP_est)
dist_t = np.sqrt((X_t[:,8]-X_t[:,0])**2 + (X_t[:,9]-X_t[:,1])**2)
dist_e = np.sqrt((X_e[:,8]-X_e[:,0])**2 + (X_e[:,9]-X_e[:,1])**2)

fig, axes = plt.subplots(4, 2, figsize=(18, 18))
axes[0,0].plot(X_t[:,0], X_t[:,1], 'k--', label="D1 Réel"); axes[0,0].plot(X_e[:,0], X_e[:,1], 'b-', label="D1 Est")
axes[0,0].plot(X_t[:,8], X_t[:,9], 'k--', label="D2 Réel"); axes[0,0].plot(X_e[:,8], X_e[:,9], 'r-', label="D2 Est (Biaisé)")
axes[0,0].set_title(f"Trajectoire (GPS @ {dt_gps}s)"); axes[0,0].legend(); axes[0,0].axis('equal')

axes[0,1].plot(time, dist_t, 'k--', label="Dist Réelle"); axes[0,1].plot(time, dist_e, 'g-', label="Dist Est")
axes[0,1].set_title(f"Distance relative (Ranging: {USE_RANGING})"); axes[0,1].legend()

# On affiche les 5 variables classiques
labels = ["X (m)", "Y (m)", "Vx (m/s)", "Vy (m/s)", "Theta (rad)"]
for j in range(5):
    ax = axes[(j+2)//2, (j+2)%2]
    idx1, idx2 = j, j+8
    ax.plot(time, X_t[:, idx1], 'k--', alpha=0.5)
    ax.plot(time, X_e[:, idx1], 'b', label="D1")
    ax.plot(time, X_e[:, idx2], 'r', label="D2")
    # Couloir 3-sigma D2
    s = np.sqrt(P_e[:, idx2])
    ax.fill_between(time, X_e[:, idx2]-3*s, X_e[:, idx2]+3*s, color='red', alpha=0.1)
    ax.set_title(labels[j]); ax.grid(True)

plt.tight_layout()
plt.savefig(f"ekf_expert_ranging_{USE_RANGING}_gps_{dt_gps}.png", dpi=300)
plt.show()

