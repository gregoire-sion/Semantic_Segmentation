import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np # Au cas où tes données sont en .npy

# ==========================================
# 1. ARCHITECTURE DU KALMANNET
# ==========================================

class KalmanGainNetwork(nn.Module):
    def __init__(self, state_dim, obs_dim, hidden_dim=64):
        super(KalmanGainNetwork, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, state_dim * obs_dim)

    def forward(self, features, hidden_state):
        out, hidden_state = self.gru(features, hidden_state)
        out = self.fc(out)
        K = out.view(-1, self.state_dim, self.obs_dim)
        return K, hidden_state

class EKF_KalmanNet(nn.Module):
    def __init__(self, state_dim, obs_dim, f_func, h_func):
        super(EKF_KalmanNet, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.f = f_func
        self.h = h_func
        
        self.kalman_net = KalmanGainNetwork(state_dim, obs_dim)

    def forward(self, y_seq):
        batch_size, seq_len, _ = y_seq.shape
        device = y_seq.device
        
        x_hat = torch.zeros(batch_size, self.state_dim, device=device)
        h_gru = torch.zeros(1, batch_size, self.kalman_net.gru.hidden_size, device=device)
        
        x_hat_seq = []

        for t in range(seq_len):
            y_t = y_seq[:, t, :]
            
            # Prédiction EKF
            x_pred = self.f(x_hat) 
            y_pred = self.h(x_pred)
            
            # Innovation
            innovation = y_t - y_pred 
            
            # Calcul du Gain K
            features = innovation.unsqueeze(1)
            K_t, h_gru = self.kalman_net(features, h_gru)
            
            # Mise à jour
            innovation_unsqueezed = innovation.unsqueeze(2) 
            update_term = torch.bmm(K_t, innovation_unsqueezed).squeeze(2)
            
            x_hat = x_pred + update_term
            x_hat_seq.append(x_hat)

        return torch.stack(x_hat_seq, dim=1)

# ==========================================
# 2. BOUCLE D'ENTRAÎNEMENT
# ==========================================

def train_kalman_net(model, dataloader, epochs=50, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    print(f"Début de l'entraînement sur {device}...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_obs, batch_true_states in dataloader:
            batch_obs, batch_true_states = batch_obs.to(device), batch_true_states.to(device)
            
            optimizer.zero_grad()
            estimated_states = model(batch_obs)
            loss = criterion(estimated_states, batch_true_states)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | MSE Loss: {avg_loss:.6f}")
            
    print("Entraînement terminé !")
    return model

# ==========================================
# 3. EXÉCUTION : CHARGEMENT ET ENTRAÎNEMENT
# ==========================================

if __name__ == "__main__":
    # --- A REMPLIR AVEC TES PROPRES FONCTIONS PHYSIQUES ---
    def f_systeme(x):
        # Ta fonction de transition d'état f(x)
        return x 

    def h_systeme(x):
        # Ta fonction d'observation h(x)
        return x 

    # --- 1. CHARGEMENT DE TON DATASET ---
    print("Chargement des données...")
    
    # Option A : Si tu as sauvegardé tes données avec torch.save()
    observations_tensor = torch.load("chemin/vers/tes_observations.pt") # Shape: [N, seq_len, obs_dim]
    true_states_tensor = torch.load("chemin/vers/tes_etats_vrais.pt")   # Shape: [N, seq_len, state_dim]
    
    # Option B : Si tu as sauvegardé en Numpy (.npy)
    # obs_np = np.load("chemin/vers/tes_observations.npy")
    # states_np = np.load("chemin/vers/tes_etats_vrais.npy")
    # observations_tensor = torch.tensor(obs_np, dtype=torch.float32)
    # true_states_tensor = torch.tensor(states_np, dtype=torch.float32)

    # Paramètres déduits automatiquement de tes données
    BATCH_SIZE = 32 # A ajuster selon ta RAM/VRAM
    STATE_DIM = true_states_tensor.shape[-1]
    OBS_DIM = observations_tensor.shape[-1]
    EPOCHS = 100
    
    dataset = TensorDataset(observations_tensor, true_states_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 2. INSTANCIATION DU MODÈLE ---
    model = EKF_KalmanNet(
        state_dim=STATE_DIM, 
        obs_dim=OBS_DIM, 
        f_func=f_systeme, 
        h_func=h_systeme
    )

    # --- 3. LANCEMENT DE L'ENTRAÎNEMENT ---
    trained_model = train_kalman_net(model, dataloader, epochs=EPOCHS, lr=1e-3)
    
    # Sauvegarde du modèle entraîné (optionnel mais recommandé)
    torch.save(trained_model.state_dict(), "kalmannet_weights.pth")
    print("Modèle sauvegardé sous 'kalmannet_weights.pth'")


import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PARAMÈTRES GLOBAUX
# ==========================================
dt_imu = 0.01   # 100 Hz
dt_gps = 0.5    # GPS lent (0.5 Hz) pour créer de gros sauts
dt_uwb = 0.1    # Capteur de distance rapide (10 Hz)
sim_time = 30.0 

# ==========================================
# 🔴 LES SWITCHS (SCÉNARIO MODULABLE) 🔴
# ==========================================
USE_RANGING = True  # Active le capteur de distance entre les drones

# Capteurs Drone 1 (Le "Maître" bien équipé)
USE_GPS_D1 = True
USE_VEL_D1 = True

# Capteurs Drone 2 (Le "Suiveur" aveugle)
USE_GPS_D2 = False
USE_VEL_D2 = False

step_gps = int(dt_gps / dt_imu)
step_uwb = int(dt_uwb / dt_imu)
N_steps = int(sim_time / dt_imu)
time = np.arange(0, sim_time, dt_imu)

# ==========================================
# CLASSE SWARM EKF (Modulable)
# ==========================================
class SwarmEKF_Baseline:
    def __init__(self, start_y2=10.0):
        # Bruits standard
        q = np.diag([0.01, 0.01, 0.1, 0.1, np.deg2rad(1.0)]) ** 2
        self.Q = np.block([[q, np.zeros((5,5))], [np.zeros((5,5)), q]])
        
        # Bruits de mesure (Base 2x2, assemblés dynamiquement)
        self.R_gps = np.diag([3.0, 3.0]) ** 2
        self.R_vel = np.diag([0.5, 0.5]) ** 2 
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

    def update_gps(self, use_d1, use_d2):
        if not use_d1 and not use_d2:
            return # Aucun drone n'a de GPS, on passe
            
        H_list, Z_list, R_list = [], [], []
        
        if use_d1:
            Z_list.append(self.X_true[0:2] + np.random.normal(0, np.sqrt(self.R_gps[0,0]), (2,1)))
            H1 = np.zeros((2, 10)); H1[0, 0] = 1; H1[1, 1] = 1
            H_list.append(H1)
            R_list.extend([self.R_gps[0,0], self.R_gps[1,1]])
            
        if use_d2:
            Z_list.append(self.X_true[5:7] + np.random.normal(0, np.sqrt(self.R_gps[0,0]), (2,1)))
            H2 = np.zeros((2, 10)); H2[0, 5] = 1; H2[1, 6] = 1
            H_list.append(H2)
            R_list.extend([self.R_gps[0,0], self.R_gps[1,1]])
            
        H = np.vstack(H_list)
        Z = np.vstack(Z_list)
        R = np.diag(R_list)
        
        y = Z - (H @ self.X_est)
        S = H @ self.P_est @ H.T + R
        K = self.P_est @ H.T @ np.linalg.inv(S)
        
        self.X_est = self.X_est + K @ y
        self.P_est = (np.eye(10) - K @ H) @ self.P_est

    def update_velocity(self, use_d1, use_d2):
        if not use_d1 and not use_d2:
            return
            
        H_list, Z_list, R_list = [], [], []
        
        if use_d1:
            Z_list.append(self.X_true[2:4] + np.random.normal(0, np.sqrt(self.R_vel[0,0]), (2,1)))
            H1 = np.zeros((2, 10)); H1[0, 2] = 1; H1[1, 3] = 1
            H_list.append(H1)
            R_list.extend([self.R_vel[0,0], self.R_vel[1,1]])
            
        if use_d2:
            Z_list.append(self.X_true[7:9] + np.random.normal(0, np.sqrt(self.R_vel[0,0]), (2,1)))
            H2 = np.zeros((2, 10)); H2[0, 7] = 1; H2[1, 8] = 1
            H_list.append(H2)
            R_list.extend([self.R_vel[0,0], self.R_vel[1,1]])
            
        H = np.vstack(H_list)
        Z = np.vstack(Z_list)
        R = np.diag(R_list)
        
        y = Z - (H @ self.X_est)
        S = H @ self.P_est @ H.T + R
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
        swarm.update_gps(USE_GPS_D1, USE_GPS_D2)
        swarm.update_velocity(USE_VEL_D1, USE_VEL_D2)
        
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
# AFFICHAGE PROPRE
# ==========================================
fig = plt.figure(figsize=(16, 18))
plt.subplots_adjust(hspace=0.4)

état_txt = f"UWB:{USE_RANGING} | D1(GPS:{USE_GPS_D1}, VEL:{USE_VEL_D1}) | D2(GPS:{USE_GPS_D2}, VEL:{USE_VEL_D2})"

# --- 1. Trajectoire 2D ---
plt.subplot(4, 2, 1)
plt.plot(X_t[:,0], X_t[:,1], 'k--', label="D1 Réel")
plt.plot(X_e[:,0], X_e[:,1], 'b-', label="D1 Est")
plt.plot(X_t[:,5], X_t[:,6], 'k:', linewidth=2, label="D2 Réel")
plt.plot(X_e[:,5], X_e[:,6], 'r-', label="D2 Est")
plt.title(f"Trajectoire 2D\n({état_txt})"); plt.xlabel("X (m)"); plt.ylabel("Y (m)")
plt.legend(); plt.grid(True); plt.axis('equal')

# --- 2. Distance ---
plt.subplot(4, 2, 2)
plt.plot(time, dist_t, 'k--', label="Distance Réelle")
plt.plot(time, dist_e, 'g-', label="Distance Estimée")
plt.title("Distance D1-D2"); plt.xlabel("Temps (s)"); plt.ylabel("Distance (m)")
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
    
    # Drone 2 (Vérité et Estimé)
    plt.plot(time, X_t[:, j+5], 'k:', alpha=0.7, label="D2 Réel" if j==0 else "")
    plt.plot(time, X_e[:, j+5], 'r-', label="D2 Est" if j==0 else "")
    s2 = np.sqrt(P_e[:, j+5])
    plt.fill_between(time, X_e[:, j+5]-3*s2, X_e[:, j+5]+3*s2, color='red', alpha=0.1)

    plt.title(labels[j]); plt.xlabel("Temps (s)"); plt.grid(True)
    if j == 0: plt.legend(loc="upper left")

# Sauvegarde avec un nom de fichier dynamique selon la configuration
filename = f"figures/EKF_2_drones_merged/baseline_uwb_{USE_RANGING}_D1_{USE_GPS_D1}{USE_VEL_D1}_D2_{USE_GPS_D2}{USE_VEL_D2}.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Terminé ! Image sauvegardée : {filename}")


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. PARAMÈTRES & CONFIGURATION GPU
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Appareil utilisé : {device.type.upper()}")

BATCH_SIZE = 5000
SIM_TIME = 20.0         
DT_IMU = 0.01           
DT_GPS = 1.0            
DT_UWB = 0.1            

STEPS_PER_TRAJ = int(SIM_TIME / DT_IMU)
STEP_GPS = int(DT_GPS / DT_IMU)
STEP_UWB = int(DT_UWB / DT_IMU)

# ==========================================
# 2. INITIALISATION DES TENSEURS (SUR GPU)
# ==========================================
Q_base_vals = [0.01, 0.01, 0.1, 0.1, np.deg2rad(1.0)]
Q_base = torch.diag(torch.tensor(Q_base_vals, dtype=torch.float32, device=device)) ** 2
Q_full = torch.block_diag(Q_base, Q_base).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)

R_gps = (torch.diag(torch.tensor([3.0, 3.0, 3.0, 3.0], dtype=torch.float32, device=device)) ** 2).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
R_uwb = torch.tensor([[[0.5 ** 2]]], dtype=torch.float32, device=device).repeat(BATCH_SIZE, 1, 1)

X_true = torch.zeros((BATCH_SIZE, 10, 1), device=device)
X_est = torch.zeros((BATCH_SIZE, 10, 1), device=device)

start_y2 = torch.empty((BATCH_SIZE, 1, 1), device=device).uniform_(5.0, 15.0)
X_true[:, 6:7, :] = start_y2
X_est[:, 6:7, :] = start_y2

P_est = torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) * 5.0

bias_d2 = torch.empty((BATCH_SIZE, 3, 1), device=device).uniform_(-0.2, 0.2)

H_gps = torch.zeros((BATCH_SIZE, 4, 10), device=device)
H_gps[:, 0, 0] = 1; H_gps[:, 1, 1] = 1; H_gps[:, 2, 5] = 1; H_gps[:, 3, 6] = 1

# ==========================================
# 3. STOCKAGE DE L'HISTORIQUE
# ==========================================
hist_features = torch.zeros((BATCH_SIZE, STEPS_PER_TRAJ, 25), dtype=torch.float32)

current_ax = torch.zeros((BATCH_SIZE, 1), device=device)
current_omega = torch.zeros((BATCH_SIZE, 1), device=device)

# ==========================================
# 4. LA BOUCLE TEMPORELLE BATCHÉE
# ==========================================
print(f"⚙️ Simulation de {BATCH_SIZE} trajectoires sur {STEPS_PER_TRAJ} itérations...")

for step in tqdm(range(STEPS_PER_TRAJ)):
    
    if step % 400 == 0:
        current_ax.uniform_(0.0, 1.0)
        current_omega.uniform_(-0.5, 0.5)
        current_ay = torch.zeros_like(current_ax)
        
    for i in [0, 5]: 
        theta_t = X_true[:, i+4, 0:1]
        X_true[:, i, 0:1]   += X_true[:, i+2, 0:1] * DT_IMU
        X_true[:, i+1, 0:1] += X_true[:, i+3, 0:1] * DT_IMU
        X_true[:, i+2, 0:1] += (current_ax * torch.cos(theta_t) - current_ay * torch.sin(theta_t)) * DT_IMU
        X_true[:, i+3, 0:1] += (current_ax * torch.sin(theta_t) + current_ay * torch.cos(theta_t)) * DT_IMU
        X_true[:, i+4, 0:1] += current_omega * DT_IMU
        
    u_base = torch.cat([current_ax.unsqueeze(-1), current_ay.unsqueeze(-1), current_omega.unsqueeze(-1)], dim=1) 
    u1 = u_base + torch.randn((BATCH_SIZE, 3, 1), device=device) * 0.1
    u2 = u_base + bias_d2 + torch.randn((BATCH_SIZE, 3, 1), device=device) * 0.1

    X_prev_est = X_est.clone()
    X_pred = torch.zeros_like(X_est)
    F = torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
    
    for idx, u in zip([0, 5], [u1, u2]):
        x, y, vx, vy, theta_e = X_est[:, idx, 0:1], X_est[:, idx+1, 0:1], X_est[:, idx+2, 0:1], X_est[:, idx+3, 0:1], X_est[:, idx+4, 0:1]
        a_x, a_y, om = u[:, 0, :], u[:, 1, :], u[:, 2, :]
        
        X_pred[:, idx, 0:1]   = x + vx * DT_IMU
        X_pred[:, idx+1, 0:1] = y + vy * DT_IMU
        X_pred[:, idx+2, 0:1] = vx + (a_x * torch.cos(theta_e) - a_y * torch.sin(theta_e)) * DT_IMU
        X_pred[:, idx+3, 0:1] = vy + (a_x * torch.sin(theta_e) + a_y * torch.cos(theta_e)) * DT_IMU
        X_pred[:, idx+4, 0:1] = theta_e + om * DT_IMU
        
        F[:, idx, idx+2] = DT_IMU; F[:, idx+1, idx+3] = DT_IMU
        F[:, idx+2, idx+4] = (-a_x * torch.sin(theta_e) - a_y * torch.cos(theta_e)).squeeze(-1) * DT_IMU
        F[:, idx+3, idx+4] = ( a_x * torch.cos(theta_e) - a_y * torch.sin(theta_e)).squeeze(-1) * DT_IMU
        
    P_pred = torch.bmm(F, torch.bmm(P_est, F.mT)) + Q_full
    X_est = X_pred.clone()
    delta_x = X_est - X_prev_est

    has_gps, has_uwb = 0.0, 0.0
    y_gps = torch.zeros((BATCH_SIZE, 4, 1), device=device)
    y_uwb = torch.zeros((BATCH_SIZE, 1, 1), device=device)
    
    # --- GPS ---
    if step % STEP_GPS == 0:
        has_gps = 1.0
        z_gps = torch.cat([X_true[:, 0:2, :], X_true[:, 5:7, :]], dim=1) + torch.randn((BATCH_SIZE, 4, 1), device=device) * 3.0
        y_gps = z_gps - torch.bmm(H_gps, X_est)
        
        S = torch.bmm(H_gps, torch.bmm(P_pred, H_gps.mT)) + R_gps
        K = torch.bmm(P_pred, torch.bmm(H_gps.mT, torch.linalg.inv(S)))
        X_est = X_est + torch.bmm(K, y_gps)
        P_pred = torch.bmm(torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) - torch.bmm(K, H_gps), P_pred)
        
    # --- UWB ---
    if step % STEP_UWB == 0:
        has_uwb = 1.0
        dx_true = X_true[:, 5, 0] - X_true[:, 0, 0]
        dy_true = X_true[:, 6, 0] - X_true[:, 1, 0]
        true_dist = torch.sqrt(dx_true**2 + dy_true**2).unsqueeze(-1).unsqueeze(-1)
        z_dist = true_dist + torch.randn((BATCH_SIZE, 1, 1), device=device) * 0.5
        
        dx_est = X_est[:, 5, 0] - X_est[:, 0, 0]
        dy_est = X_est[:, 6, 0] - X_est[:, 1, 0]
        e_dist = torch.sqrt(dx_est**2 + dy_est**2).unsqueeze(-1).unsqueeze(-1)
        
        y_uwb = z_dist - e_dist
        
        H_dist = torch.zeros((BATCH_SIZE, 1, 10), device=device)
        
        safe_dist = torch.clamp(e_dist.squeeze(), min=0.01) 
        
        H_dist[:, 0, 0] = -dx_est / safe_dist
        H_dist[:, 0, 1] = -dy_est / safe_dist
        H_dist[:, 0, 5] = dx_est / safe_dist
        H_dist[:, 0, 6] = dy_est / safe_dist
        
        S = torch.bmm(H_dist, torch.bmm(P_pred, H_dist.mT)) + R_uwb
        K = torch.bmm(P_pred, torch.bmm(H_dist.mT, torch.linalg.inv(S)))
        X_est = X_est + torch.bmm(K, y_uwb)
        P_pred = torch.bmm(torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) - torch.bmm(K, H_dist), P_pred)

    P_est = P_pred

    step_data = torch.cat([
        torch.full((BATCH_SIZE, 1), has_gps, dtype=torch.float32),
        torch.full((BATCH_SIZE, 1), has_uwb, dtype=torch.float32),
        y_gps.squeeze(-1).cpu(),
        y_uwb.squeeze(-1).cpu(),
        delta_x.squeeze(-1).cpu(),
        X_pred[:, [0,1,5,6], 0].cpu(), 
        X_true[:, [0,1,5,6], 0].cpu()  
    ], dim=1)
    
    hist_features[:, step, :] = step_data

# ==========================================
# 5. FORMATAGE DATAFRAME ET SAUVEGARDE
# ==========================================
print("💾 Conversion du tenseur 3D en DataFrame Pandas...")

traj_ids = torch.arange(BATCH_SIZE).view(-1, 1).repeat(1, STEPS_PER_TRAJ).view(-1, 1).numpy()
time_steps = torch.arange(STEPS_PER_TRAJ).view(1, -1).repeat(BATCH_SIZE, 1).view(-1, 1).numpy()

flat_features = hist_features.view(-1, 25).numpy()

final_array = np.hstack((traj_ids, time_steps, flat_features))

columns = [
    'traj_id', 'time_step', 'has_gps', 'has_uwb',
    'y_gps_x1', 'y_gps_y1', 'y_gps_x2', 'y_gps_y2', 'y_uwb_dist',
    'dx_0', 'dx_1', 'dx_2', 'dx_3', 'dx_4', 'dx_5', 'dx_6', 'dx_7', 'dx_8', 'dx_9',
    'prior_x1', 'prior_y1', 'prior_x2', 'prior_y2',
    'true_x1', 'true_y1', 'true_x2', 'true_y2'
]

df = pd.DataFrame(final_array, columns=columns)

os.makedirs("data", exist_ok=True)

# MODIFICATION ICI : On utilise Pickle au lieu de Parquet
pickle_path = "data/kalman_dataset_gpu_.pkl"
df.to_pickle(pickle_path)

print(f"✅ Terminé ! Dataset sauvegardé : {pickle_path}")
print(f"📊 Taille finale : {len(df)} lignes ({BATCH_SIZE} trajectoires)")
