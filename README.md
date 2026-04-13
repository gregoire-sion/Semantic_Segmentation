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
