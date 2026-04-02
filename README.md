import numpy as np
import pandas as pd
import os
from tqdm import tqdm # Pour la barre de progression

# ==========================================
# 1. PARAMÈTRES DU DATASET
# ==========================================
NUM_TRAJECTORIES = 50   # Mets 1000 ou 5000 pour ton entraînement final
SIM_TIME = 20.0         # 20 secondes par trajectoire
DT_IMU = 0.01           # 100 Hz
DT_GPS = 1.0            # 1 Hz
DT_UWB = 0.1            # 10 Hz

STEPS_PER_TRAJ = int(SIM_TIME / DT_IMU)
STEP_GPS = int(DT_GPS / DT_IMU)
STEP_UWB = int(DT_UWB / DT_IMU)

# Bruits de base
Q_base = np.diag([0.01, 0.01, 0.1, 0.1, np.deg2rad(1.0)]) ** 2
Q_full = np.block([[Q_base, np.zeros((5,5))], [np.zeros((5,5)), Q_base]])
R_gps = np.diag([3.0, 3.0, 3.0, 3.0]) ** 2
R_uwb = np.array([[0.5 ** 2]])

# ==========================================
# 2. GÉNÉRATEUR DE DONNÉES
# ==========================================
def generate_trajectory(traj_id):
    """Génère un épisode complet de vol et retourne une liste de dictionnaires"""
    
    # --- Initialisation Aléatoire ---
    start_y2 = np.random.uniform(5.0, 15.0) # Le Drone 2 commence à une distance aléatoire
    X_true = np.zeros((10, 1))
    X_true[6, 0] = start_y2
    
    X_est = np.zeros((10, 1))
    X_est[6, 0] = start_y2
    P_est = np.eye(10) * 5.0
    
    # Biais IMU aléatoire pour cet épisode (entre -0.2 et +0.2)
    bias_d2 = np.random.uniform(-0.2, 0.2, (3, 1))
    
    # Consignes de vol dynamiques
    current_ax, current_ay, current_omega = 0.5, 0.0, 0.0
    
    trajectory_data = []
    
    for step in range(STEPS_PER_TRAJ):
        t = step * DT_IMU
        
        # --- 1. Changement de trajectoire aléatoire (toutes les 4 secondes) ---
        if step % 400 == 0:
            current_ax = np.random.uniform(0.0, 1.0)
            current_omega = np.random.uniform(-0.5, 0.5) # Virage gauche ou droite
            
        # --- 2. Mise à jour de la Vérité Terrain ---
        for i in [0, 5]: 
            theta = X_true[i+4, 0]
            X_true[i, 0]   += X_true[i+2, 0] * DT_IMU
            X_true[i+1, 0] += X_true[i+3, 0] * DT_IMU
            X_true[i+2, 0] += (current_ax * np.cos(theta) - current_ay * np.sin(theta)) * DT_IMU
            X_true[i+3, 0] += (current_ax * np.sin(theta) + current_ay * np.cos(theta)) * DT_IMU
            X_true[i+4, 0] += current_omega * DT_IMU
            
        # --- 3. Génération des IMU (avec Biais sur D2) ---
        u1 = np.array([[current_ax], [current_ay], [current_omega]]) + np.random.normal(0, 0.1, (3,1))
        u2 = np.array([[current_ax], [current_ay], [current_omega]]) + bias_d2 + np.random.normal(0, 0.1, (3,1))
        
        # --- 4. Prédiction EKF (L'état Prior) ---
        X_prev_est = X_est.copy()
        X_pred = np.zeros((10, 1))
        F = np.eye(10)
        for idx, u in zip([0, 5], [u1, u2]):
            x, y, vx, vy, theta = X_est[idx:idx+5, 0]
            a_x, a_y, om = u[0,0], u[1,0], u[2,0]
            
            X_pred[idx, 0]   = x + vx * DT_IMU
            X_pred[idx+1, 0] = y + vy * DT_IMU
            X_pred[idx+2, 0] = vx + (a_x * np.cos(theta) - a_y * np.sin(theta)) * DT_IMU
            X_pred[idx+3, 0] = vy + (a_x * np.sin(theta) + a_y * np.cos(theta)) * DT_IMU
            X_pred[idx+4, 0] = theta + om * DT_IMU
            
            F[idx,   idx+2] = DT_IMU; F[idx+1, idx+3] = DT_IMU
            F[idx+2, idx+4] = (-a_x * np.sin(theta) - a_y * np.cos(theta)) * DT_IMU
            F[idx+3, idx+4] = ( a_x * np.cos(theta) - a_y * np.sin(theta)) * DT_IMU
            
        P_pred = F @ P_est @ F.T + Q_full
        X_est = X_pred.copy()
        
        # Calcul du Delta X (Évolution de l'état)
        delta_x = X_est - X_prev_est

        # --- 5. Observations (Innovations et Flags) ---
        has_gps, has_uwb = 0, 0
        y_gps = np.zeros((4, 1))
        y_uwb = np.zeros((1, 1))
        
        # Correction GPS
        if step % STEP_GPS == 0:
            has_gps = 1
            z_gps = np.vstack((X_true[0:2], X_true[5:7])) + np.random.normal(0, 3.0, (4,1))
            H_gps = np.zeros((4, 10))
            H_gps[0, 0] = 1; H_gps[1, 1] = 1; H_gps[2, 5] = 1; H_gps[3, 6] = 1
            y_gps = z_gps - (H_gps @ X_est)
            # MAJ de l'EKF classique (pour continuer la boucle)
            S = H_gps @ P_pred @ H_gps.T + R_gps
            K = P_pred @ H_gps.T @ np.linalg.inv(S)
            X_est = X_est + K @ y_gps
            P_pred = (np.eye(10) - K @ H_gps) @ P_pred
            
        # Correction UWB
        if step % STEP_UWB == 0:
            has_uwb = 1
            true_dist = np.sqrt((X_true[5,0]-X_true[0,0])**2 + (X_true[6,0]-X_true[1,0])**2)
            z_dist = true_dist + np.random.normal(0, 0.5)
            e_dist = np.sqrt((X_est[5,0]-X_est[0,0])**2 + (X_est[6,0]-X_est[1,0])**2)
            y_uwb[0,0] = z_dist - e_dist
            
            H_dist = np.zeros((1, 10))
            if e_dist > 0.01:
                H_dist[0, 0] = -(X_est[5,0]-X_est[0,0]) / e_dist
                H_dist[0, 1] = -(X_est[6,0]-X_est[1,0]) / e_dist
                H_dist[0, 5] =  (X_est[5,0]-X_est[0,0]) / e_dist
                H_dist[0, 6] =  (X_est[6,0]-X_est[1,0]) / e_dist
            
            S = H_dist @ P_pred @ H_dist.T + R_uwb
            K = P_pred @ H_dist.T @ np.linalg.inv(S)
            X_est = X_est + K * y_uwb[0,0]
            P_pred = (np.eye(10) - K @ H_dist) @ P_pred
            
        P_est = P_pred # Fin du step

        # --- 6. Enregistrement de la ligne de données ---
        row = {
            'traj_id': traj_id,
            'time_step': step,
            'has_gps': has_gps,
            'has_uwb': has_uwb,
            
            # --- LES INPUTS DU RÉSEAU (Features) ---
            'y_gps_x1': y_gps[0,0], 'y_gps_y1': y_gps[1,0],
            'y_gps_x2': y_gps[2,0], 'y_gps_y2': y_gps[3,0],
            'y_uwb_dist': y_uwb[0,0],
        }
        
        # Ajout des 10 variables de delta_x
        for j in range(10): row[f'dx_{j}'] = delta_x[j, 0]
            
        # --- LES TARGETS DU RÉSEAU (Pour la Loss) ---
        # On sauvegarde l'état Prédit (avant correction) et la Vérité Absolue
        for j in range(10): 
            row[f'prior_{j}'] = X_pred[j, 0]
            row[f'true_{j}'] = X_true[j, 0]
            
        trajectory_data.append(row)
        
    return trajectory_data

# ==========================================
# 3. CRÉATION ET SAUVEGARDE
# ==========================================
print(f"🚀 Génération du Dataset ({NUM_TRAJECTORIES} trajectoires)...")
all_data = []

for traj_idx in tqdm(range(NUM_TRAJECTORIES)):
    traj_data = generate_trajectory(traj_idx)
    all_data.extend(traj_data)

# Conversion en DataFrame Pandas
df = pd.DataFrame(all_data)

# Sauvegarde
os.makedirs("dataset", exist_ok=True)
csv_path = "dataset/kalman_dataset.csv"
parquet_path = "dataset/kalman_dataset.parquet"

print("💾 Sauvegarde en cours...")
df.to_csv(csv_path, index=False)
try:
    df.to_parquet(parquet_path, engine='pyarrow', index=False)
    print(f"✅ Succès ! Fichier Parquet créé : {parquet_path} (Ultra-rapide pour PyTorch)")
except ImportError:
    print("⚠️ Module 'pyarrow' ou 'fastparquet' non installé. Seul le CSV a été créé.")

print(f"✅ Fichier CSV créé : {csv_path} (Idéal pour DataWrangler)")
print(f"📊 Taille du Dataset : {len(df)} lignes x {len(df.columns)} colonnes.")
print("\nAperçu des colonnes :", list(df.columns)[:10], "...")
