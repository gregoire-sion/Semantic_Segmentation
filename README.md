import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. PARAMÈTRES & CONFIGURATION GPU
# ==========================================
# Vérification du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Appareil utilisé : {device.type.upper()}")

BATCH_SIZE = 500        # Nombre de trajectoires simulées EN MÊME TEMPS
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
# CORRECTION ICI : On force le dtype=torch.float32 pour éviter la contamination par np.deg2rad
Q_base_vals = [0.01, 0.01, 0.1, 0.1, np.deg2rad(1.0)]
Q_base = torch.diag(torch.tensor(Q_base_vals, dtype=torch.float32, device=device)) ** 2
Q_full = torch.block_diag(Q_base, Q_base).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) # Shape: (B, 10, 10)

# On sécurise aussi les autres tenseurs constants au cas où
R_gps = (torch.diag(torch.tensor([3.0, 3.0, 3.0, 3.0], dtype=torch.float32, device=device)) ** 2).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
R_uwb = torch.tensor([[[0.5 ** 2]]], dtype=torch.float32, device=device).repeat(BATCH_SIZE, 1, 1)

# État Réel et Estimé (Shape: Batch, 10 variables, 1 colonne)
X_true = torch.zeros((BATCH_SIZE, 10, 1), device=device)
X_est = torch.zeros((BATCH_SIZE, 10, 1), device=device)

# Le Drone 2 commence à une distance aléatoire Y (entre 5m et 15m) pour chaque trajectoire
start_y2 = torch.empty((BATCH_SIZE, 1, 1), device=device).uniform_(5.0, 15.0)
X_true[:, 6:7, :] = start_y2
X_est[:, 6:7, :] = start_y2

# Matrice de Covariance initiale P (Shape: B, 10, 10)
P_est = torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) * 5.0

# Biais IMU aléatoire par trajectoire (Shape: B, 3, 1)
bias_d2 = torch.empty((BATCH_SIZE, 3, 1), device=device).uniform_(-0.2, 0.2)

# Matrices d'Observation Constantes
H_gps = torch.zeros((BATCH_SIZE, 4, 10), device=device)
H_gps[:, 0, 0] = 1; H_gps[:, 1, 1] = 1; H_gps[:, 2, 5] = 1; H_gps[:, 3, 6] = 1

# ==========================================
# 3. STOCKAGE DE L'HISTORIQUE (SUR CPU POUR LA RAM)
# ==========================================
# On pré-alloue l'espace sur la RAM standard pour ne pas exploser la VRAM du GPU
hist_features = torch.zeros((BATCH_SIZE, STEPS_PER_TRAJ, 25), dtype=torch.float32)

current_ax = torch.zeros((BATCH_SIZE, 1), device=device)
current_omega = torch.zeros((BATCH_SIZE, 1), device=device)

# ==========================================
# 4. LA BOUCLE TEMPORELLE BATCHÉE
# ==========================================
print(f"⚙️ Simulation de {BATCH_SIZE} trajectoires sur {STEPS_PER_TRAJ} itérations...")

for step in tqdm(range(STEPS_PER_TRAJ)):
    
    # 1. Changement de consigne aléatoire (toutes les 4 sec)
    if step % 400 == 0:
        current_ax.uniform_(0.0, 1.0)
        current_omega.uniform_(-0.5, 0.5)
        current_ay = torch.zeros_like(current_ax) # Pas de glissement latéral
        
    # 2. Mise à jour de la Vérité (Opérations vectorielles sur les batchs d'un coup)
    for i in [0, 5]: 
        theta_t = X_true[:, i+4, 0:1]
        X_true[:, i, 0:1]   += X_true[:, i+2, 0:1] * DT_IMU
        X_true[:, i+1, 0:1] += X_true[:, i+3, 0:1] * DT_IMU
        X_true[:, i+2, 0:1] += (current_ax * torch.cos(theta_t) - current_ay * torch.sin(theta_t)) * DT_IMU
        X_true[:, i+3, 0:1] += (current_ax * torch.sin(theta_t) + current_ay * torch.cos(theta_t)) * DT_IMU
        X_true[:, i+4, 0:1] += current_omega * DT_IMU
        
    # 3. Mesures IMU Bruitées
    u_base = torch.cat([current_ax.unsqueeze(-1), current_ay.unsqueeze(-1), current_omega.unsqueeze(-1)], dim=1) 
    u1 = u_base + torch.randn((BATCH_SIZE, 3, 1), device=device) * 0.1
    u2 = u_base + bias_d2 + torch.randn((BATCH_SIZE, 3, 1), device=device) * 0.1

    # 4. Prédiction EKF (L'état Prior)
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

    # 5. Corrections (Innovations)
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
        safe_dist = torch.clamp(e_dist.squeeze(-1), min=0.01) # Éviter division par zéro
        H_dist[:, 0, 0] = -dx_est / safe_dist
        H_dist[:, 0, 1] = -dy_est / safe_dist
        H_dist[:, 0, 5] = dx_est / safe_dist
        H_dist[:, 0, 6] = dy_est / safe_dist
        
        S = torch.bmm(H_dist, torch.bmm(P_pred, H_dist.mT)) + R_uwb
        K = torch.bmm(P_pred, torch.bmm(H_dist.mT, torch.linalg.inv(S)))
        X_est = X_est + torch.bmm(K, y_uwb)
        P_pred = torch.bmm(torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) - torch.bmm(K, H_dist), P_pred)

    P_est = P_pred

    # 6. Enregistrement dans le CPU (Vectorisé)
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

# Création des index (Traj_ID, Time_Step)
traj_ids = torch.arange(BATCH_SIZE).view(-1, 1).repeat(1, STEPS_PER_TRAJ).view(-1, 1).numpy()
time_steps = torch.arange(STEPS_PER_TRAJ).view(1, -1).repeat(BATCH_SIZE, 1).view(-1, 1).numpy()

# Aplatissement du tenseur
flat_features = hist_features.view(-1, 25).numpy()

# Fusion des identifiants et des données
final_array = np.hstack((traj_ids, time_steps, flat_features))

columns = [
    'traj_id', 'time_step', 'has_gps', 'has_uwb',
    'y_gps_x1', 'y_gps_y1', 'y_gps_x2', 'y_gps_y2', 'y_uwb_dist',
    'dx_0', 'dx_1', 'dx_2', 'dx_3', 'dx_4', 'dx_5', 'dx_6', 'dx_7', 'dx_8', 'dx_9',
    'prior_x1', 'prior_y1', 'prior_x2', 'prior_y2',
    'true_x1', 'true_y1', 'true_x2', 'true_y2'
]

df = pd.DataFrame(final_array, columns=columns)

os.makedirs("dataset/data", exist_ok=True)
parquet_path = "dataset/data/kalman_dataset_gpu.parquet"

df.to_parquet(parquet_path, engine='pyarrow', index=False)
print(f"✅ Terminé ! Dataset sauvegardé : {parquet_path}")
print(f"📊 Taille finale : {len(df)} lignes ({BATCH_SIZE} trajectoires)")




import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ==========================================
# 1. PARAMÈTRES & CONFIGURATION GPU
# ==========================================
# Vérification du GPU (C'est ici que ça active nvidia-smi !)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Appareil utilisé : {device.type.upper()}")

BATCH_SIZE = 500 #5000 initalement      # Nombre de trajectoires simulées EN MÊME TEMPS
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
# Les bruits de base (Tenseurs constants)
Q_base = torch.diag(torch.tensor([0.01, 0.01, 0.1, 0.1, np.deg2rad(1.0)], device=device)) ** 2
Q_full = torch.block_diag(Q_base, Q_base).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) # Shape: (B, 10, 10)

R_gps = (torch.diag(torch.tensor([3.0, 3.0, 3.0, 3.0], device=device)) ** 2).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
R_uwb = torch.tensor([[[0.5 ** 2]]], device=device).repeat(BATCH_SIZE, 1, 1)

# État Réel et Estimé (Shape: Batch, 10 variables, 1 colonne)
X_true = torch.zeros((BATCH_SIZE, 10, 1), device=device)
X_est = torch.zeros((BATCH_SIZE, 10, 1), device=device)

# Le Drone 2 commence à une distance aléatoire Y (entre 5m et 15m) pour chaque trajectoire
start_y2 = torch.empty((BATCH_SIZE, 1, 1), device=device).uniform_(5.0, 15.0)
X_true[:, 6:7, :] = start_y2
X_est[:, 6:7, :] = start_y2

# Matrice de Covariance initiale P (Shape: B, 10, 10)
P_est = torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) * 5.0

# Biais IMU aléatoire par trajectoire (Shape: B, 3, 1)
bias_d2 = torch.empty((BATCH_SIZE, 3, 1), device=device).uniform_(-0.2, 0.2)

# Matrices d'Observation Constantes
H_gps = torch.zeros((BATCH_SIZE, 4, 10), device=device)
H_gps[:, 0, 0] = 1; H_gps[:, 1, 1] = 1; H_gps[:, 2, 5] = 1; H_gps[:, 3, 6] = 1

# ==========================================
# 3. STOCKAGE DE L'HISTORIQUE (SUR CPU POUR LA RAM)
# ==========================================
# On pré-alloue l'espace sur la RAM standard pour ne pas exploser la VRAM du GPU
hist_features = torch.zeros((BATCH_SIZE, STEPS_PER_TRAJ, 25), dtype=torch.float32)

current_ax = torch.zeros((BATCH_SIZE, 1), device=device)
current_omega = torch.zeros((BATCH_SIZE, 1), device=device)

# ==========================================
# 4. LA BOUCLE TEMPORELLE BATCHÉE
# ==========================================
print(f"⚙️ Simulation de {BATCH_SIZE} trajectoires sur {STEPS_PER_TRAJ} itérations...")

for step in tqdm(range(STEPS_PER_TRAJ)):
    
    # 1. Changement de consigne aléatoire (toutes les 4 sec)
    if step % 400 == 0:
        current_ax.uniform_(0.0, 1.0)
        current_omega.uniform_(-0.5, 0.5)
        current_ay = torch.zeros_like(current_ax) # Pas de glissement latéral
        
    # 2. Mise à jour de la Vérité (Opérations vectorielles sur les 5000 d'un coup)
    for i in [0, 5]: 
        theta_t = X_true[:, i+4, 0:1]
        X_true[:, i, 0:1]   += X_true[:, i+2, 0:1] * DT_IMU
        X_true[:, i+1, 0:1] += X_true[:, i+3, 0:1] * DT_IMU
        X_true[:, i+2, 0:1] += (current_ax * torch.cos(theta_t) - current_ay * torch.sin(theta_t)) * DT_IMU
        X_true[:, i+3, 0:1] += (current_ax * torch.sin(theta_t) + current_ay * torch.cos(theta_t)) * DT_IMU
        X_true[:, i+4, 0:1] += current_omega * DT_IMU
        
    # 3. Mesures IMU Bruitées
    u_base = torch.cat([current_ax.unsqueeze(-1), current_ay.unsqueeze(-1), current_omega.unsqueeze(-1)], dim=1) # Shape: (B, 3, 1)
    u1 = u_base + torch.randn((BATCH_SIZE, 3, 1), device=device) * 0.1
    u2 = u_base + bias_d2 + torch.randn((BATCH_SIZE, 3, 1), device=device) * 0.1

    # 4. Prédiction EKF (L'état Prior)
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
        F[:, idx+2, idx+4] = (-a_x * torch.sin(theta_e) - a_y * torch.cos(theta_e)).squeeze() * DT_IMU
        F[:, idx+3, idx+4] = ( a_x * torch.cos(theta_e) - a_y * torch.sin(theta_e)).squeeze() * DT_IMU
        
    P_pred = torch.bmm(F, torch.bmm(P_est, F.mT)) + Q_full
    X_est = X_pred.clone()
    delta_x = X_est - X_prev_est

    # 5. Corrections (Innovations)
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
        safe_dist = torch.clamp(e_dist.squeeze(), min=0.01) # Éviter division par zéro
        H_dist[:, 0, 0] = -dx_est / safe_dist
        H_dist[:, 0, 1] = -dy_est / safe_dist
        H_dist[:, 0, 5] = dx_est / safe_dist
        H_dist[:, 0, 6] = dy_est / safe_dist
        
        S = torch.bmm(H_dist, torch.bmm(P_pred, H_dist.mT)) + R_uwb
        K = torch.bmm(P_pred, torch.bmm(H_dist.mT, torch.linalg.inv(S)))
        X_est = X_est + torch.bmm(K, y_uwb)
        P_pred = torch.bmm(torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) - torch.bmm(K, H_dist), P_pred)

    P_est = P_pred

    # 6. Enregistrement dans le CPU (Vectorisé)
    # [has_gps, has_uwb, y_gps(4), y_uwb(1), delta_x(10), prior(4), true(4)] = 25 colonnes
    # On stocke seulement les positions (X,Y) dans Prior et True pour alléger le dataset
    step_data = torch.cat([
        torch.full((BATCH_SIZE, 1), has_gps),
        torch.full((BATCH_SIZE, 1), has_uwb),
        y_gps.squeeze(-1).cpu(),
        y_uwb.squeeze(-1).cpu(),
        delta_x.squeeze(-1).cpu(),
        X_pred[:, [0,1,5,6], 0].cpu(), # Prior X, Y (D1 et D2)
        X_true[:, [0,1,5,6], 0].cpu()  # True X, Y (D1 et D2)
    ], dim=1)
    
    hist_features[:, step, :] = step_data

# ==========================================
# 5. FORMATAGE DATAFRAME ET SAUVEGARDE
# ==========================================
print("💾 Conversion du tenseur 3D en DataFrame Pandas...")

# Création des index (Traj_ID, Time_Step)
traj_ids = torch.arange(BATCH_SIZE).view(-1, 1).repeat(1, STEPS_PER_TRAJ).view(-1, 1).numpy()
time_steps = torch.arange(STEPS_PER_TRAJ).view(1, -1).repeat(BATCH_SIZE, 1).view(-1, 1).numpy()

# Aplatissement du tenseur (Batch * Steps, Features)
flat_features = hist_features.view(-1, 25).numpy()

# Fusion des identifiants et des données
final_array = np.hstack((traj_ids, time_steps, flat_features))

columns = [
    'traj_id', 'time_step', 'has_gps', 'has_uwb',
    'y_gps_x1', 'y_gps_y1', 'y_gps_x2', 'y_gps_y2', 'y_uwb_dist',
    'dx_0', 'dx_1', 'dx_2', 'dx_3', 'dx_4', 'dx_5', 'dx_6', 'dx_7', 'dx_8', 'dx_9',
    'prior_x1', 'prior_y1', 'prior_x2', 'prior_y2',
    'true_x1', 'true_y1', 'true_x2', 'true_y2'
]

df = pd.DataFrame(final_array, columns=columns)

os.makedirs("dataset", exist_ok=True)
parquet_path = "dataset/data/kalman_dataset_gpu.parquet"

# On sauvegarde directement en Parquet (Le CSV serait trop lourd : ~3 millions de lignes !)
df.to_parquet(parquet_path, engine='pyarrow', index=False)
print(f"✅ Terminé ! Dataset sauvegardé : {parquet_path}")
print(f"📊 Taille finale : {len(df)} lignes ({BATCH_SIZE} trajectoires)")

(Environnement) scvmpr10.fr.mbda.priv:/home/gsionsua/Work_bis/GRU/dataset $python generate_dataset.py 
🚀 Appareil utilisé : CUDA
⚙️ Simulation de 500 trajectoires sur 2000 itérations...
  0%|                                                                                                 | 0/2000 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/gsionsua/Work_bis/GRU/dataset/generate_dataset.py", line 126, in <module>
    S = torch.bmm(H_gps, torch.bmm(P_pred, H_gps.mT)) + R_gps
RuntimeError: expected scalar type Float but found Double


-------------------------------------------

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
filename = f"baseline_uwb_{USE_RANGING}_D1_{USE_GPS_D1}{USE_VEL_D1}_D2_{USE_GPS_D2}{USE_VEL_D2}.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Terminé ! Image sauvegardée : {filename}")


import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PARAMÈTRES GLOBAUX
# ==========================================
dt_imu = 0.01   # 100 Hz
dt_gps = 0.5    # GPS lent (0.5 Hz) pour créer de gros sauts
dt_uwb = 0.1    # Capteur de distance rapide (10 Hz)
sim_time = 30.0 

# 🔴 LE SWITCH 🔴
USE_RANGING = False

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

plt.savefig(f"figures/EKF_2_drones_merged/baseline_simple_ranging_{USE_RANGING}_gps_{dt_gps}.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Terminé ! Image sauvegardée : baseline_simple_ranging_{USE_RANGING}_gps_{dt_gps}.png")

python -c "import torch; print('\n✅ Import réussi !'); print(f'🎮 GPU disponible : {torch.cuda.is_available()}'); print(f'📟 Nom du GPU : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"


python -c "import torch; print('\n' + '='*25); print(f'🚀 PyTorch Version: {torch.__version__}'); print(f'🎮 GPU Disponible: {torch.cuda.is_available()}'); print('='*25)"


env_GPU) scvmpr12.fr.mbda.priv:/home/gsionsua/Work_bis/GRU/dataset $rm -rf ~/miniconda3
rm: impossible de supprimer '/home/gsionsua/miniconda3/envs/env_GPU/lib': Le dossier n'est pas vide
ERROR: ld.so: object '/home/gsionsua/miniconda3/envs/env_GPU/lib/libmkl_rt.so' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.


(env_GPU) scvmpr12.fr.mbda.priv:/home/gsionsua/Work_bis/GRU/dataset $export LD_PRELOAD=$CONDA_PREFIX/lib/libmkl_rt.so python -c "import torch; print('Succes'); print(f'GPU: {torch.cuda.is_available()}')"
bash: export: « -c » : identifiant non valable
bash: export: « import torch; print('Succes'); print(f'GPU: {torch.cuda.is_available()}') » : identifiant non valable


export LD_PRELOAD=$CONDA_PREFIX/lib/libmkl_rt.so
python -c "import torch; print('Succes'); print(f'GPU: {torch.cuda.is_available()}')"


(env_GPU) scvmpr12.fr.mbda.priv:/home/gsionsua/Work_bis/GRU/dataset $python -c "
> import torch; 
> import pandas; 
> import tqdm; 
> print('\n' + '='*20);
> print(f'🔥 PyTorch GPU : {torch.cuda.is_available()}');
> print(f'📊 Pandas OK   : {pandas.__version__}');
> print(f'🚀 TQDM OK     : Prêt');
> print('='*20);
> "
Traceback (most recent call last):
  File "<string>", line 2, in <module>
  File "/home/gsionsua/miniconda3/envs/env_GPU/lib/python3.11/site-packages/torch/__init__.py", line 367, in <module>
    from torch._C import *  # noqa: F403
    ^^^^^^^^^^^^^^^^^^^^^^
ImportError: /home/gsionsua/miniconda3/envs/env_GPU/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
(env_GPU) scvmpr12.fr.mbda.priv:/home/gsionsua/Work_bis/GRU/dataset $export LD_PRELOAD=$HOME/miniconda3/envs/env_GPU/lib/libmkl_rt.so:$HOME/miniconda3/envs/env_GPU/lib/libintel_thunder.so.5
ERROR: ld.so: object '/home/gsionsua/miniconda3/envs/env_GPU/lib/libintel_thunder.so.5' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
(env_GPU) scvmpr12.fr.mbda.priv:/home/gsionsua/Work_bis/GRU/dataset $python -c "import torch; print('✅ Succès !'); print(f'GPU: {torch.cuda.is_available()}')"
bash: !': event not found
(env_GPU) scvmpr12.fr.mbda.priv:/home/gsionsua/Work_bis/GRU/dataset $
ERROR: ld.so: object '/home/gsionsua/miniconda3/envs/env_GPU/lib/libintel_thunder.so.5' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
(env_GPU) scvmpr12.fr.mbda.priv:/home/gsionsua/Work_bis/GRU/dataset $export LD_PRELOAD=$HOME/miniconda3/envs/env_GPU/lib/libmkl_rt.so:$HOME/miniconda3/envs/env_GPU/lib/libintel_thunder.so.5
ERROR: ld.so: object '/home/gsionsua/miniconda3/envs/env_GPU/lib/libintel_thunder.so.5' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
(env_GPU) scvmpr12.fr.mbda.priv:/home/gsionsua/Work_bis/GRU/dataset $
ERROR: ld.so: object '/home/gsionsua/minico

export LD_PRELOAD=$HOME/miniconda3/envs/env_GPU/lib/libmkl_rt.so:$HOME/miniconda3/envs/env_GPU/lib/libintel_thunder.so.5
python -c "import torch; print('✅ Succès !'); print(f'GPU: {torch.cuda.is_available()}')"


python -c "
import torch; 
import pandas; 
import tqdm; 
print('\n' + '='*20);
print(f'🔥 PyTorch GPU : {torch.cuda.is_available()}');
print(f'📊 Pandas OK   : {pandas.__version__}');
print(f'🚀 TQDM OK     : Prêt');
print('='*20);
"


python -c "import torch; print(f'🚀 GPU disponible : {torch.cuda.is_available()}'); print(f'📟 Carte : {torch.cuda.get_device_name(0)}')"



python -c "import torch; print('\n✅ Import réussi !'); print(f'🔥 PyTorch version: {torch.__version__}'); print(f'🎮 GPU disponible: {torch.cuda.is_available()}'); print(f'📟 Nom du GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')"


(env_GPU) scvmpr12.fr.mbda.priv:/home/gsionsua/Work_bis/GRU/dataset $python test_env.py 
Traceback (most recent call last):
  File "/home/gsionsua/Work_bis/GRU/dataset/test_env.py", line 1, in <module>
    import torch
  File "/home/gsionsua/miniconda3/envs/env_GPU/lib/python3.11/site-packages/torch/__init__.py", line 367, in <module>
    from torch._C import *  # noqa: F403
    ^^^^^^^^^^^^^^^^^^^^^^
ImportError: /home/gsionsua/miniconda3/envs/env_GPU/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent



export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
python test_gpu.py

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


ldd ~/miniconda3/envs/drone_gpu/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so | grep "not found"


import torch
import sys
import time

def run_diagnostic():
    print("="*30)
    print("🔍 DIAGNOSTIC ENVIRONNEMENT")
    print("="*30)
    
    # 1. Version de Python et PyTorch
    print(f"🐍 Python : {sys.version.split()[0]}")
    print(f"🔥 PyTorch : {torch.__version__}")
    
    # 2. Vérification CUDA
    cuda_disponible = torch.cuda.is_available()
    print(f"🎮 CUDA disponible : {'OUI ✅' if cuda_disponible else 'NON ❌'}")
    
    if cuda_disponible:
        # 3. Infos sur le GPU et Driver
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        cuda_version = torch.version.cuda
        print(f"📟 Nom du GPU : {gpu_name}")
        print(f"🛠️ Version CUDA de PyTorch : {cuda_version}")
        
        # 4. Test de calcul réel (Multiplication de matrices)
        print("\n🚀 Lancement du test de calcul...")
        try:
            start_time = time.time()
            # On crée deux grosses matrices directement sur le GPU
            a = torch.randn(5000, 5000).cuda()
            b = torch.randn(5000, 5000).cuda()
            
            # Multiplication matricielle (très gourmand en GPU)
            c = torch.matmul(a, b)
            
            # On attend que le GPU finisse (asynchrone par défaut)
            torch.cuda.synchronize()
            duration = time.time() - start_time
            
            print(f"✅ Test réussi en {duration:.4f} secondes !")
            print(f"📦 Mémoire GPU utilisée : {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
        except Exception as e:
            print(f"💥 Erreur lors du calcul : {e}")
    else:
        print("\n⚠️ ATTENTION : Le GPU n'est pas détecté.")
        print("Vérifie que ton driver NVIDIA est bien chargé avec 'nvidia-smi'.")

    print("="*30)

if __name__ == "__main__":
    run_diagnostic()



python -c "import torch; print(f'Statut : GPU OK' if torch.cuda.is_available() else 'Erreur : Driver non reconnu'); print(f'Version CUDA interne : {torch.version.cuda}')"


python -c "import torch; import pandas; print('--- CHECK-UP ---'); print(f'PyTorch OK (v{torch.__version__})'); print(f'GPU visible: {torch.cuda.is_available()}'); print(f'Pandas OK (v{pandas.__version__})')"


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ==========================================
# 1. PARAMÈTRES & CONFIGURATION GPU
# ==========================================
# Vérification du GPU (C'est ici que ça active nvidia-smi !)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Appareil utilisé : {device.type.upper()}")

BATCH_SIZE = 500 #5000 initalement      # Nombre de trajectoires simulées EN MÊME TEMPS
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
# Les bruits de base (Tenseurs constants)
Q_base = torch.diag(torch.tensor([0.01, 0.01, 0.1, 0.1, np.deg2rad(1.0)], device=device)) ** 2
Q_full = torch.block_diag(Q_base, Q_base).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) # Shape: (B, 10, 10)

R_gps = (torch.diag(torch.tensor([3.0, 3.0, 3.0, 3.0], device=device)) ** 2).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
R_uwb = torch.tensor([[[0.5 ** 2]]], device=device).repeat(BATCH_SIZE, 1, 1)

# État Réel et Estimé (Shape: Batch, 10 variables, 1 colonne)
X_true = torch.zeros((BATCH_SIZE, 10, 1), device=device)
X_est = torch.zeros((BATCH_SIZE, 10, 1), device=device)

# Le Drone 2 commence à une distance aléatoire Y (entre 5m et 15m) pour chaque trajectoire
start_y2 = torch.empty((BATCH_SIZE, 1, 1), device=device).uniform_(5.0, 15.0)
X_true[:, 6:7, :] = start_y2
X_est[:, 6:7, :] = start_y2

# Matrice de Covariance initiale P (Shape: B, 10, 10)
P_est = torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) * 5.0

# Biais IMU aléatoire par trajectoire (Shape: B, 3, 1)
bias_d2 = torch.empty((BATCH_SIZE, 3, 1), device=device).uniform_(-0.2, 0.2)

# Matrices d'Observation Constantes
H_gps = torch.zeros((BATCH_SIZE, 4, 10), device=device)
H_gps[:, 0, 0] = 1; H_gps[:, 1, 1] = 1; H_gps[:, 2, 5] = 1; H_gps[:, 3, 6] = 1

# ==========================================
# 3. STOCKAGE DE L'HISTORIQUE (SUR CPU POUR LA RAM)
# ==========================================
# On pré-alloue l'espace sur la RAM standard pour ne pas exploser la VRAM du GPU
hist_features = torch.zeros((BATCH_SIZE, STEPS_PER_TRAJ, 25), dtype=torch.float32)

current_ax = torch.zeros((BATCH_SIZE, 1), device=device)
current_omega = torch.zeros((BATCH_SIZE, 1), device=device)

# ==========================================
# 4. LA BOUCLE TEMPORELLE BATCHÉE
# ==========================================
print(f"⚙️ Simulation de {BATCH_SIZE} trajectoires sur {STEPS_PER_TRAJ} itérations...")

for step in tqdm(range(STEPS_PER_TRAJ)):
    
    # 1. Changement de consigne aléatoire (toutes les 4 sec)
    if step % 400 == 0:
        current_ax.uniform_(0.0, 1.0)
        current_omega.uniform_(-0.5, 0.5)
        current_ay = torch.zeros_like(current_ax) # Pas de glissement latéral
        
    # 2. Mise à jour de la Vérité (Opérations vectorielles sur les 5000 d'un coup)
    for i in [0, 5]: 
        theta_t = X_true[:, i+4, 0:1]
        X_true[:, i, 0:1]   += X_true[:, i+2, 0:1] * DT_IMU
        X_true[:, i+1, 0:1] += X_true[:, i+3, 0:1] * DT_IMU
        X_true[:, i+2, 0:1] += (current_ax * torch.cos(theta_t) - current_ay * torch.sin(theta_t)) * DT_IMU
        X_true[:, i+3, 0:1] += (current_ax * torch.sin(theta_t) + current_ay * torch.cos(theta_t)) * DT_IMU
        X_true[:, i+4, 0:1] += current_omega * DT_IMU
        
    # 3. Mesures IMU Bruitées
    u_base = torch.cat([current_ax.unsqueeze(-1), current_ay.unsqueeze(-1), current_omega.unsqueeze(-1)], dim=1) # Shape: (B, 3, 1)
    u1 = u_base + torch.randn((BATCH_SIZE, 3, 1), device=device) * 0.1
    u2 = u_base + bias_d2 + torch.randn((BATCH_SIZE, 3, 1), device=device) * 0.1

    # 4. Prédiction EKF (L'état Prior)
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
        F[:, idx+2, idx+4] = (-a_x * torch.sin(theta_e) - a_y * torch.cos(theta_e)).squeeze() * DT_IMU
        F[:, idx+3, idx+4] = ( a_x * torch.cos(theta_e) - a_y * torch.sin(theta_e)).squeeze() * DT_IMU
        
    P_pred = torch.bmm(F, torch.bmm(P_est, F.mT)) + Q_full
    X_est = X_pred.clone()
    delta_x = X_est - X_prev_est

    # 5. Corrections (Innovations)
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
        safe_dist = torch.clamp(e_dist.squeeze(), min=0.01) # Éviter division par zéro
        H_dist[:, 0, 0] = -dx_est / safe_dist
        H_dist[:, 0, 1] = -dy_est / safe_dist
        H_dist[:, 0, 5] = dx_est / safe_dist
        H_dist[:, 0, 6] = dy_est / safe_dist
        
        S = torch.bmm(H_dist, torch.bmm(P_pred, H_dist.mT)) + R_uwb
        K = torch.bmm(P_pred, torch.bmm(H_dist.mT, torch.linalg.inv(S)))
        X_est = X_est + torch.bmm(K, y_uwb)
        P_pred = torch.bmm(torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) - torch.bmm(K, H_dist), P_pred)

    P_est = P_pred

    # 6. Enregistrement dans le CPU (Vectorisé)
    # [has_gps, has_uwb, y_gps(4), y_uwb(1), delta_x(10), prior(4), true(4)] = 25 colonnes
    # On stocke seulement les positions (X,Y) dans Prior et True pour alléger le dataset
    step_data = torch.cat([
        torch.full((BATCH_SIZE, 1), has_gps),
        torch.full((BATCH_SIZE, 1), has_uwb),
        y_gps.squeeze(-1).cpu(),
        y_uwb.squeeze(-1).cpu(),
        delta_x.squeeze(-1).cpu(),
        X_pred[:, [0,1,5,6], 0].cpu(), # Prior X, Y (D1 et D2)
        X_true[:, [0,1,5,6], 0].cpu()  # True X, Y (D1 et D2)
    ], dim=1)
    
    hist_features[:, step, :] = step_data

# ==========================================
# 5. FORMATAGE DATAFRAME ET SAUVEGARDE
# ==========================================
print("💾 Conversion du tenseur 3D en DataFrame Pandas...")

# Création des index (Traj_ID, Time_Step)
traj_ids = torch.arange(BATCH_SIZE).view(-1, 1).repeat(1, STEPS_PER_TRAJ).view(-1, 1).numpy()
time_steps = torch.arange(STEPS_PER_TRAJ).view(1, -1).repeat(BATCH_SIZE, 1).view(-1, 1).numpy()

# Aplatissement du tenseur (Batch * Steps, Features)
flat_features = hist_features.view(-1, 25).numpy()

# Fusion des identifiants et des données
final_array = np.hstack((traj_ids, time_steps, flat_features))

columns = [
    'traj_id', 'time_step', 'has_gps', 'has_uwb',
    'y_gps_x1', 'y_gps_y1', 'y_gps_x2', 'y_gps_y2', 'y_uwb_dist',
    'dx_0', 'dx_1', 'dx_2', 'dx_3', 'dx_4', 'dx_5', 'dx_6', 'dx_7', 'dx_8', 'dx_9',
    'prior_x1', 'prior_y1', 'prior_x2', 'prior_y2',
    'true_x1', 'true_y1', 'true_x2', 'true_y2'
]

df = pd.DataFrame(final_array, columns=columns)

os.makedirs("dataset", exist_ok=True)
parquet_path = "dataset/data/kalman_dataset_gpu.parquet"

# On sauvegarde directement en Parquet (Le CSV serait trop lourd : ~3 millions de lignes !)
df.to_parquet(parquet_path, engine='pyarrow', index=False)
print(f"✅ Terminé ! Dataset sauvegardé : {parquet_path}")
print(f"📊 Taille finale : {len(df)} lignes ({BATCH_SIZE} trajectoires)")
