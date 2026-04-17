    P_pred = torch.bmm(F, torch.bmm(P_est, F.mT)) + Q_full
RuntimeError: The size of tensor a (10) must match the size of tensor b (5) at non-singleton dimension 2


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

BATCH_SIZE = 5000       # Nombre de trajectoires
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
# --- VARIABILITÉ DU BRUIT --- 
# On génère des niveaux de bruit différents pour chaque trajectoire du batch
# Cela aide le KalmanNet à apprendre à estimer le gain K de manière adaptative.
noise_scale_Q = torch.empty((BATCH_SIZE, 1, 1), device=device).uniform_(0.5, 2.0)
noise_scale_R = torch.empty((BATCH_SIZE, 1, 1), device=device).uniform_(0.5, 2.0)

Q_base_vals = [0.01, 0.01, 0.1, 0.1, np.deg2rad(1.0)]
Q_base = torch.diag(torch.tensor(Q_base_vals, dtype=torch.float32, device=device)) ** 2
Q_full = Q_base.unsqueeze(0).repeat(BATCH_SIZE, 1, 1) * noise_scale_Q

R_gps_base = (torch.diag(torch.tensor([3.0, 3.0, 3.0, 3.0], dtype=torch.float32, device=device)) ** 2)
R_gps = R_gps_base.unsqueeze(0).repeat(BATCH_SIZE, 1, 1) * noise_scale_R

R_uwb_base = torch.tensor([[[0.5 ** 2]]], dtype=torch.float32, device=device)
R_uwb = R_uwb_base.repeat(BATCH_SIZE, 1, 1) * noise_scale_R

# États
X_true = torch.zeros((BATCH_SIZE, 10, 1), device=device)
X_est = torch.zeros((BATCH_SIZE, 10, 1), device=device)

# Position initiale aléatoire pour le Drone 2
start_y2 = torch.empty((BATCH_SIZE, 1, 1), device=device).uniform_(5.0, 15.0)
X_true[:, 6:7, :] = start_y2
X_est[:, 6:7, :] = start_y2

P_est = torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) * 5.0

# Biais IMU Drone 2 (le "suiveur" est défectueux)
bias_d2 = torch.empty((BATCH_SIZE, 3, 1), device=device).uniform_(-0.1, 0.1)

H_gps = torch.zeros((BATCH_SIZE, 4, 10), device=device)
H_gps[:, 0, 0] = 1; H_gps[:, 1, 1] = 1; H_gps[:, 2, 5] = 1; H_gps[:, 3, 6] = 1

# Historique (25 colonnes pour correspondre au script d'entraînement)
hist_features = torch.zeros((BATCH_SIZE, STEPS_PER_TRAJ, 25), dtype=torch.float32)

current_ax = torch.zeros((BATCH_SIZE, 1), device=device)
current_omega = torch.zeros((BATCH_SIZE, 1), device=device)

# ==========================================
# 3. LA BOUCLE TEMPORELLE BATCHÉE
# ==========================================
print(f"⚙️ Simulation et Génération du Dataset...")

for step in tqdm(range(STEPS_PER_TRAJ)):
    
    # Changement de dynamique aléatoire toutes les 4s
    if step % 400 == 0:
        current_ax.uniform_(0.1, 0.8)
        current_omega.uniform_(-0.4, 0.4)
        current_ay = torch.zeros_like(current_ax)
        
    # --- VÉRITÉ TERRAIN (Modèle Non-Linéaire f) ---
    for i in [0, 5]: 
        theta_t = X_true[:, i+4, 0:1]
        # x_next = x + vx*dt, y_next = y + vy*dt
        X_true[:, i, 0:1]   += X_true[:, i+2, 0:1] * DT_IMU
        X_true[:, i+1, 0:1] += X_true[:, i+3, 0:1] * DT_IMU
        # vx_next = vx + (ax*cos - ay*sin)*dt ...
        X_true[:, i+2, 0:1] += (current_ax * torch.cos(theta_t) - current_ay * torch.sin(theta_t)) * DT_IMU
        X_true[:, i+3, 0:1] += (current_ax * torch.sin(theta_t) + current_ay * torch.cos(theta_t)) * DT_IMU
        X_true[:, i+4, 0:1] += current_omega * DT_IMU
        
    # --- MESURES IMU BRUITÉES ---
    u_base = torch.cat([current_ax.unsqueeze(-1), current_ay.unsqueeze(-1), current_omega.unsqueeze(-1)], dim=1) 
    u1 = u_base + torch.randn((BATCH_SIZE, 3, 1), device=device) * 0.05
    u2 = u_base + bias_d2 + torch.randn((BATCH_SIZE, 3, 1), device=device) * 0.05

    # --- PRÉDICTION EKF (Linéarisation F) ---
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
        
        # Remplissage de la Jacobienne F (Non-linéarité Commande x État)
        F[:, idx, idx+2] = DT_IMU; F[:, idx+1, idx+3] = DT_IMU
        F[:, idx+2, idx+4] = (-a_x * torch.sin(theta_e) - a_y * torch.cos(theta_e)).squeeze(-1) * DT_IMU
        F[:, idx+3, idx+4] = ( a_x * torch.cos(theta_e) - a_y * torch.sin(theta_e)).squeeze(-1) * DT_IMU
        
    # Propagation P_pred (On garde la Baseline EKF pour générer les features)
    P_pred = torch.bmm(F, torch.bmm(P_est, F.mT)) + Q_full
    X_est = X_pred.clone()
    delta_x = X_est - X_prev_est # Le dx_t utilisé par le KalmanNet

    has_gps, has_uwb = 0.0, 0.0
    y_gps = torch.zeros((BATCH_SIZE, 4, 1), device=device)
    y_uwb = torch.zeros((BATCH_SIZE, 1, 1), device=device)
    
    # --- UPDATE GPS ---
    if step % STEP_GPS == 0:
        has_gps = 1.0
        noise_gps = torch.randn((BATCH_SIZE, 4, 1), device=device) * torch.sqrt(noise_scale_R) * 3.0
        z_gps = torch.cat([X_true[:, 0:2, :], X_true[:, 5:7, :]], dim=1) + noise_gps
        y_gps = z_gps - torch.bmm(H_gps, X_est) # Innovation y
        
        S = torch.bmm(H_gps, torch.bmm(P_pred, H_gps.mT)) + R_gps
        K = torch.bmm(P_pred, torch.bmm(H_gps.mT, torch.linalg.inv(S)))
        X_est = X_est + torch.bmm(K, y_gps)
        P_pred = torch.bmm(torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) - torch.bmm(K, H_gps), P_pred)
        
    # --- UPDATE UWB (Distance Non-linéaire) ---
    if step % STEP_UWB == 0:
        has_uwb = 1.0
        dx_t = X_true[:, 5, 0] - X_true[:, 0, 0]
        dy_t = X_true[:, 6, 0] - X_true[:, 1, 0]
        true_dist = torch.sqrt(dx_t**2 + dy_t**2).unsqueeze(-1).unsqueeze(-1)
        z_dist = true_dist + torch.randn((BATCH_SIZE, 1, 1), device=device) * torch.sqrt(noise_scale_R) * 0.5
        
        dx_e = X_est[:, 5, 0] - X_est[:, 0, 0]
        dy_e = X_est[:, 6, 0] - X_est[:, 1, 0]
        e_dist = torch.sqrt(dx_e**2 + dy_e**2).unsqueeze(-1).unsqueeze(-1)
        y_uwb = z_dist - e_dist
        
        H_dist = torch.zeros((BATCH_SIZE, 1, 10), device=device)
        safe_dist = torch.clamp(e_dist.squeeze(), min=0.01) 
        H_dist[:, 0, 0] = -dx_e / safe_dist; H_dist[:, 0, 1] = -dy_e / safe_dist
        H_dist[:, 0, 5] = dx_e / safe_dist; H_dist[:, 0, 6] = dy_e / safe_dist
        
        S = torch.bmm(H_dist, torch.bmm(P_pred, H_dist.mT)) + R_uwb
        K = torch.bmm(P_pred, torch.bmm(H_dist.mT, torch.linalg.inv(S)))
        X_est = X_est + torch.bmm(K, y_uwb)
        P_pred = torch.bmm(torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) - torch.bmm(K, H_dist), P_pred)

    P_est = P_pred

    # --- STOCKAGE HISTORIQUE ---
    # Colonnes 0-24 : correspond exactement à la structure attendue par ton KalmanNet
    step_data = torch.cat([
        torch.full((BATCH_SIZE, 1), has_gps, dtype=torch.float32),
        torch.full((BATCH_SIZE, 1), has_uwb, dtype=torch.float32),
        y_gps.squeeze(-1).cpu(),    # 4 dim
        y_uwb.squeeze(-1).cpu(),    # 1 dim
        delta_x.squeeze(-1).cpu(),  # 10 dim
        X_pred[:, [0,1,5,6], 0].cpu(), # prior positions (4 dim)
        X_true[:, [0,1,5,6], 0].cpu()  # true positions (4 dim)
    ], dim=1)
    
    hist_features[:, step, :] = step_data

# ==========================================
# 4. SAUVEGARDE
# ==========================================
print("💾 Conversion et Sauvegarde (Pickle)...")

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
df.to_pickle("data/kalman_dataset_gpu_.pkl")

print(f"✅ Terminé ! Dataset prêt pour l'entraînement optimal.")


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
print(f"Appareil utilisé : {device.type.upper()}")

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
print("Conversion du tenseur 3D en DataFrame Pandas...")

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

print(f"Terminé ! Dataset sauvegardé : {pickle_path}")
print(f"Taille finale : {len(df)} lignes ({BATCH_SIZE} trajectoires)")
