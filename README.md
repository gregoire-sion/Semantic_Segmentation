import pandas as pd
import plotly.graph_objects as go
import os

def generate_interactive_plot(data_path="data/kalman_dataset_test.pkl", traj_id=0):
    print(f"📂 Chargement des données depuis {data_path}...")
    try:
        df = pd.read_pickle(data_path)
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier {data_path} est introuvable.")
        return

    # 1. Filtrage sur la trajectoire choisie
    df_traj = df[df['traj_id'] == traj_id].sort_values(by='time_step').reset_index(drop=True)
    
    # Création d'une colonne temps (en secondes) pour l'affichage au survol
    df_traj['time_sec'] = df_traj['time_step'] * 0.01
    hover_template = "Temps : %{customdata:.2f} s<br>X : %{x:.2f} m<br>Y : %{y:.2f} m"

    print(f"🚁 Génération du graphique pour la trajectoire #{traj_id}...")
    fig = go.Figure()

    # ==========================================
    # TRACÉS POUR LE DRONE 1
    # ==========================================
    # 1. Vérité Terrain D1
    fig.add_trace(go.Scatter(
        x=df_traj['true_x1'], y=df_traj['true_y1'],
        mode='lines', name='Vérité Terrain (D1)',
        line=dict(color='black', width=3),
        customdata=df_traj['time_sec'],
        hovertemplate=hover_template
    ))

    # 2. Estimation EKF D1
    fig.add_trace(go.Scatter(
        x=df_traj['prior_x1'], y=df_traj['prior_y1'],
        mode='lines', name='EKF Baseline (D1)',
        line=dict(color='red', width=2, dash='dash'),
        customdata=df_traj['time_sec'],
        hovertemplate=hover_template
    ))

    # 3. Mesures GPS D1
    df_gps = df_traj[df_traj['has_gps'] == 1.0]
    fig.add_trace(go.Scatter(
        x=df_gps['prior_x1'] + df_gps['y_gps_x1'], 
        y=df_gps['prior_y1'] + df_gps['y_gps_y1'],
        mode='markers', name='Mesures GPS (D1)',
        marker=dict(color='green', size=8, symbol='cross'),
        customdata=df_gps['time_sec'],
        hovertemplate="<b>Mesure GPS</b><br>" + hover_template
    ))

    # ==========================================
    # TRACÉS POUR LE DRONE 2
    # ==========================================
    # 1. Vérité Terrain D2
    fig.add_trace(go.Scatter(
        x=df_traj['true_x2'], y=df_traj['true_y2'],
        mode='lines', name='Vérité Terrain (D2)',
        line=dict(color='gray', width=3),
        customdata=df_traj['time_sec'],
        hovertemplate=hover_template,
        visible='legendonly' # Caché par défaut pour ne pas surcharger
    ))

    # 2. Estimation EKF D2
    fig.add_trace(go.Scatter(
        x=df_traj['prior_x2'], y=df_traj['prior_y2'],
        mode='lines', name='EKF Baseline (D2)',
        line=dict(color='orange', width=2, dash='dash'),
        customdata=df_traj['time_sec'],
        hovertemplate=hover_template,
        visible='legendonly'
    ))

    # 3. Mesures GPS D2
    fig.add_trace(go.Scatter(
        x=df_gps['prior_x2'] + df_gps['y_gps_x2'], 
        y=df_gps['prior_y2'] + df_gps['y_gps_y2'],
        mode='markers', name='Mesures GPS (D2)',
        marker=dict(color='purple', size=8, symbol='cross'),
        customdata=df_gps['time_sec'],
        hovertemplate="<b>Mesure GPS</b><br>" + hover_template,
        visible='legendonly'
    ))

    # ==========================================
    # MISE EN PAGE DU GRAPHIQUE
    # ==========================================
    fig.update_layout(
        title=f"<b>Analyse du Dataset : Trajectoire #{traj_id}</b><br><sup>Comparaison de la vérité terrain, de l'EKF et des capteurs</sup>",
        xaxis_title="Position X (mètres)",
        yaxis_title="Position Y (mètres)",
        yaxis=dict(scaleanchor="x", scaleratio=1), # Très important : 1m en X = 1m en Y à l'écran
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            title="Cliquez sur un élément pour l'afficher/masquer :",
            yanchor="top", y=0.99, 
            xanchor="left", x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="Black",
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    # ==========================================
    # SAUVEGARDE ET AFFICHAGE
    # ==========================================
    output_filename = f"trajectoire_{traj_id}_interactive.html"
    fig.write_html(output_filename)
    print(f"✅ Fichier HTML généré avec succès : {output_filename}")
    
    # Ouvre automatiquement le graphique dans ton navigateur
    fig.show()

if __name__ == "__main__":
    # Tu peux changer le numéro de la trajectoire ici (de 0 à 499)
    generate_interactive_plot(traj_id=0)


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 1. REDÉFINITION DU MODÈLE
# ==========================================
class KalmanNet_Gain(nn.Module):
    def __init__(self, input_dim=17, obs_dim=5, state_dim=10, hidden_dim=64):
        super(KalmanNet_Gain, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.activation = nn.Tanh()
        self.fc = nn.Linear(hidden_dim, state_dim * obs_dim)

    def forward(self, features):
        x = features.transpose(1, 2)
        x = self.bn_input(x)
        x = x.transpose(1, 2)
        gru_out, _ = self.gru(x)
        gru_out = self.activation(gru_out)
        K_flat = self.fc(gru_out)
        K = K_flat.view(features.size(0), features.size(1), self.state_dim, self.obs_dim)
        return K

# ==========================================
# 2. GÉNÉRATION DE LA TRAJECTOIRE EXOTIQUE
# ==========================================
def simulate_exotic_trajectory(model_path="weights/train4/kalmannet_best_weights.pth"):
    device = torch.device("cpu") # On fait ce test sur CPU car il n'y a qu'une trajectoire
    print("🌪️ Génération de la trajectoire exotique (Torture Test)...")
    
    SIM_TIME = 20.0         
    DT_IMU = 0.01           
    DT_GPS = 1.0            
    DT_UWB = 0.1            
    STEPS = int(SIM_TIME / DT_IMU)

    # Bruits fixes pour le test
    Q_vals = [0.01, 0.01, 0.1, 0.1, np.deg2rad(1.0)]
    Q_base = torch.diag(torch.tensor(Q_vals, dtype=torch.float32)) ** 2
    Q_full = torch.block_diag(Q_base, Q_base).unsqueeze(0)
    R_gps = (torch.diag(torch.tensor([3.0, 3.0, 3.0, 3.0], dtype=torch.float32)) ** 2).unsqueeze(0)
    R_uwb = torch.tensor([[[0.5 ** 2]]], dtype=torch.float32)

    X_true = torch.zeros((1, 10, 1))
    X_est = torch.zeros((1, 10, 1))
    X_true[:, 6:7, :] = 10.0 # Y initial du Drone 2
    X_est[:, 6:7, :] = 10.0
    P_est = torch.eye(10).unsqueeze(0) * 5.0
    H_gps = torch.zeros((1, 4, 10)); H_gps[:, 0, 0] = 1; H_gps[:, 1, 1] = 1; H_gps[:, 2, 5] = 1; H_gps[:, 3, 6] = 1

    features_list = []
    y_t_list = []
    priors_list = []
    truths_list = []

    for step in range(STEPS):
        t = step * DT_IMU
        
        # --- COMMANDES EXOTIQUES ---
        # Accélération sinusoïdale et rotation folle
        current_ax = torch.tensor([[1.0 + 0.5 * np.sin(t)]])
        current_omega = torch.tensor([[0.5 * np.cos(t * 1.5)]])
        current_ay = torch.zeros((1, 1))
        
        # Rafale de vent vicieuse (non modélisée dans l'EKF) pendant la panne GPS
        if 8.0 < t < 14.0:
            wind_ay = 1.5 # Le vent pousse le drone sur le côté
        else:
            wind_ay = 0.0

        # Vérité Terrain
        for i in [0, 5]:
            theta_t = X_true[:, i+4, 0:1]
            X_true[:, i, 0:1]   += X_true[:, i+2, 0:1] * DT_IMU
            X_true[:, i+1, 0:1] += X_true[:, i+3, 0:1] * DT_IMU
            X_true[:, i+2, 0:1] += (current_ax * torch.cos(theta_t) - (current_ay + wind_ay) * torch.sin(theta_t)) * DT_IMU
            X_true[:, i+3, 0:1] += (current_ax * torch.sin(theta_t) + (current_ay + wind_ay) * torch.cos(theta_t)) * DT_IMU
            X_true[:, i+4, 0:1] += current_omega * DT_IMU

        # IMU Bruitée
        u_base = torch.cat([current_ax.unsqueeze(-1), current_ay.unsqueeze(-1), current_omega.unsqueeze(-1)], dim=1)
        u = u_base + torch.randn((1, 3, 1)) * 0.05

        # EKF Prédiction
        X_prev_est = X_est.clone()
        X_pred = torch.zeros_like(X_est)
        F = torch.eye(10).unsqueeze(0)
        
        for idx in [0, 5]:
            theta_e = X_est[:, idx+4, 0:1]
            a_x, a_y, om = u[:, 0, :], u[:, 1, :], u[:, 2, :]
            
            X_pred[:, idx, 0]   = X_est[:, idx, 0] + X_est[:, idx+2, 0] * DT_IMU
            X_pred[:, idx+1, 0] = X_est[:, idx+1, 0] + X_est[:, idx+3, 0] * DT_IMU
            X_pred[:, idx+2, 0] = X_est[:, idx+2, 0] + (a_x * torch.cos(theta_e) - a_y * torch.sin(theta_e)).squeeze(-1) * DT_IMU
            X_pred[:, idx+3, 0] = X_est[:, idx+3, 0] + (a_x * torch.sin(theta_e) + a_y * torch.cos(theta_e)).squeeze(-1) * DT_IMU
            X_pred[:, idx+4, 0] = theta_e.squeeze(-1) + om.squeeze(-1) * DT_IMU
            
            F[:, idx, idx+2] = DT_IMU; F[:, idx+1, idx+3] = DT_IMU
            F[:, idx+2, idx+4] = (-a_x * torch.sin(theta_e) - a_y * torch.cos(theta_e)).squeeze(-1) * DT_IMU
            F[:, idx+3, idx+4] = ( a_x * torch.cos(theta_e) - a_y * torch.sin(theta_e)).squeeze(-1) * DT_IMU
            
        P_pred = torch.bmm(F, torch.bmm(P_est, F.mT)) + Q_full
        X_est = X_pred.clone()
        dx = X_est - X_prev_est

        # Mises à jour Capteurs
        has_gps, has_uwb = 0.0, 0.0
        y_gps = torch.zeros((1, 4, 1))
        y_uwb = torch.zeros((1, 1, 1))
        
        # PANNE GPS MASSIVE ENTRE 8s ET 14s
        is_gps_outage = 8.0 < t < 14.0

        if step % STEP_GPS == 0 and not is_gps_outage:
            has_gps = 1.0
            z_gps = torch.cat([X_true[:, 0:2, :], X_true[:, 5:7, :]], dim=1) + torch.randn((1, 4, 1)) * 3.0
            y_gps = z_gps - torch.bmm(H_gps, X_est)
            
            S = torch.bmm(H_gps, torch.bmm(P_pred, H_gps.mT)) + R_gps
            K_ekf = torch.bmm(P_pred, torch.bmm(H_gps.mT, torch.linalg.inv(S)))
            X_est = X_est + torch.bmm(K_ekf, y_gps)
            P_pred = torch.bmm(torch.eye(10) - torch.bmm(K_ekf, H_gps), P_pred)
            
        if step % STEP_UWB == 0:
            has_uwb = 1.0
            dist_t = torch.sqrt((X_true[:, 5, 0]-X_true[:, 0, 0])**2 + (X_true[:, 6, 0]-X_true[:, 1, 0])**2).unsqueeze(-1).unsqueeze(-1)
            z_dist = dist_t + torch.randn((1, 1, 1)) * 0.5
            dx_e, dy_e = X_est[:, 5, 0]-X_est[:, 0, 0], X_est[:, 6, 0]-X_est[:, 1, 0]
            e_dist = torch.sqrt(dx_e**2 + dy_e**2).unsqueeze(-1).unsqueeze(-1)
            y_uwb = z_dist - e_dist
            
            H_dist = torch.zeros((1, 1, 10))
            safe_dist = torch.clamp(e_dist.squeeze(), min=0.01) 
            H_dist[:, 0, 0] = -dx_e / safe_dist; H_dist[:, 0, 1] = -dy_e / safe_dist
            H_dist[:, 0, 5] = dx_e / safe_dist; H_dist[:, 0, 6] = dy_e / safe_dist
            
            S = torch.bmm(H_dist, torch.bmm(P_pred, H_dist.mT)) + R_uwb
            K_ekf = torch.bmm(P_pred, torch.bmm(H_dist.mT, torch.linalg.inv(S)))
            X_est = X_est + torch.bmm(K_ekf, y_uwb)
            P_pred = torch.bmm(torch.eye(10) - torch.bmm(K_ekf, H_dist), P_pred)

        P_est = P_pred

        # Stockage pour le KalmanNet
        feat = torch.cat([torch.tensor([[has_gps]]), torch.tensor([[has_uwb]]), y_gps.squeeze(-1), y_uwb.squeeze(-1), dx.squeeze(-1)], dim=1)
        yt = torch.cat([y_gps.squeeze(-1), y_uwb.squeeze(-1)], dim=1)
        
        features_list.append(feat)
        y_t_list.append(yt)
        priors_list.append(X_pred[:, [0,1,5,6], 0])
        truths_list.append(X_true[:, [0,1,5,6], 0])

    # Empilage des séquences
    features = torch.stack(features_list, dim=1)
    y_t = torch.stack(y_t_list, dim=1).unsqueeze(-1)
    priors = torch.stack(priors_list, dim=1)
    truths = torch.stack(truths_list, dim=1)

    # --- 3. INFÉRENCE KALMANNET ---
    print("🧠 Inférence du KalmanNet...")
    model = KalmanNet_Gain()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        K_net = model(features)
        state_update = torch.matmul(K_net, y_t).squeeze(-1)
        pos_update = state_update[:, :, [0, 1, 5, 6]]
        kalmannet_est = priors + pos_update

    # --- 4. AFFICHAGE DES RÉSULTATS ---
    print("📊 Génération du graphique de la trajectoire exotique...")
    t_truths = truths.numpy()[0]
    t_priors = priors.numpy()[0]
    t_knet = kalmannet_est.numpy()[0]

    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Trace les trajectoires du Drone 1
    ax.plot(t_truths[:, 0], t_truths[:, 1], 'k-', linewidth=3, label="Vérité D1 (Avec rafale secrète)")
    ax.plot(t_priors[:, 0], t_priors[:, 1], 'r--', linewidth=2, alpha=0.8, label="EKF D1 (Dérive)")
    ax.plot(t_knet[:, 0], t_knet[:, 1], 'b-', linewidth=2, label="KalmanNet D1")
    
    # Ajout d'une zone grisée pour indiquer la panne GPS
    # On trouve approximativement où est le drone entre t=8 et t=14
    idx_start, idx_end = 800, 1400
    x_min = min(t_truths[idx_start:idx_end, 0]) - 5
    x_max = max(t_truths[idx_start:idx_end, 0]) + 5
    y_min = min(t_truths[idx_start:idx_end, 1]) - 5
    y_max = max(t_truths[idx_start:idx_end, 1]) + 5
    
    rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=1, edgecolor='none', facecolor='red', alpha=0.15, label="Zone de Panne GPS (8s - 14s)")
    ax.add_patch(rect)

    # Indiquer le point de départ
    ax.scatter(t_truths[0, 0], t_truths[0, 1], color='green', s=150, marker='*', zorder=5, label="Départ")
    
    ax.set_title("Torture Test : Panne GPS + Vent + Spirale", fontsize=16, fontweight='bold')
    ax.set_xlabel("Position X (m)")
    ax.set_ylabel("Position Y (m)")
    ax.legend(loc="upper left")
    ax.grid(True)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig("test_exotique_panne_gps.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    simulate_exotic_trajectory()


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # NOUVEAU : Pour de plus beaux graphiques statistiques

# ==========================================
# 1. REDÉFINITION DE L'ARCHITECTURE
# ==========================================
class KalmanNet_Gain(nn.Module):
    def __init__(self, input_dim=17, obs_dim=5, state_dim=10, hidden_dim=64):
        super(KalmanNet_Gain, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.activation = nn.Tanh()
        self.fc = nn.Linear(hidden_dim, state_dim * obs_dim)

    def forward(self, features):
        x = features.transpose(1, 2)
        x = self.bn_input(x)
        x = x.transpose(1, 2)
        gru_out, _ = self.gru(x)
        gru_out = self.activation(gru_out)
        K_flat = self.fc(gru_out)
        K = K_flat.view(features.size(0), features.size(1), self.state_dim, self.obs_dim)
        return K

# ==========================================
# 2. FONCTION D'ÉVALUATION ET PLOTS RICHES
# ==========================================
def evaluate_and_plot(model_path="weights/train4/kalmannet_best_weights.pth", test_data_path="data/kalman_dataset_test.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 Évaluation sur : {device.type.upper()}")

    # 1. Chargement
    model = KalmanNet_Gain().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print("Chargement du dataset de test...")
    df = pd.read_pickle(test_data_path)
    df = df.sort_values(by=['traj_id', 'time_step'])
    
    num_trajectories = df['traj_id'].nunique()
    seq_len = df['time_step'].nunique()
    
    data_np = df.drop(columns=['traj_id', 'time_step']).values
    data_tensor = torch.tensor(data_np, dtype=torch.float32).view(num_trajectories, seq_len, -1).to(device)
    
    features = data_tensor[:, :, 0:17] 
    y_t = data_tensor[:, :, 2:7].unsqueeze(-1) 
    priors = data_tensor[:, :, 17:21]   # Baseline EKF (seulement x1,y1,x2,y2)
    truths = data_tensor[:, :, 21:25]   # Vérité terrain (seulement x1,y1,x2,y2)

    # 2. Inférence
    print("Inférence en cours...")
    with torch.no_grad():
        K = model(features)
        state_update = torch.matmul(K, y_t).squeeze(-1)
        pos_update = state_update[:, :, [0, 1, 5, 6]]
        kalmannet_est = priors + pos_update 

    # 3. Calcul des Métriques
    err_baseline_d1 = torch.sqrt((priors[:,:,0]-truths[:,:,0])**2 + (priors[:,:,1]-truths[:,:,1])**2)
    err_baseline_d2 = torch.sqrt((priors[:,:,2]-truths[:,:,2])**2 + (priors[:,:,3]-truths[:,:,3])**2)
    
    err_knet_d1 = torch.sqrt((kalmannet_est[:,:,0]-truths[:,:,0])**2 + (kalmannet_est[:,:,1]-truths[:,:,1])**2)
    err_knet_d2 = torch.sqrt((kalmannet_est[:,:,2]-truths[:,:,2])**2 + (kalmannet_est[:,:,3]-truths[:,:,3])**2)
    
    rmse_baseline = torch.mean((err_baseline_d1 + err_baseline_d2) / 2.0).item()
    rmse_knet = torch.mean((err_knet_d1 + err_knet_d2) / 2.0).item()

    # --- AFFICHAGE DES STATISTIQUES ---
    print("\n" + "="*50)
    print("🎯 RÉSULTATS STATISTIQUES GLOBAUX")
    print("="*50)
    print(f"Nombre de trajectoires testées : {num_trajectories}")
    print(f"Durée par trajectoire          : {seq_len * 0.01:.1f} secondes")
    print("-" * 50)
    print(f"RMSE Baseline (EKF) : {rmse_baseline:.4f} mètres")
    print(f"RMSE KalmanNet      : {rmse_knet:.4f} mètres")
    improvement = ((rmse_baseline - rmse_knet) / rmse_baseline) * 100
    print(f"Amélioration Globale: {improvement:.2f} %")
    print("="*50 + "\n")

    # ==========================================
    # 4. GÉNÉRATION DES GRAPHIQUES RICHES
    # ==========================================
    print("Génération des graphiques...")
    sns.set_theme(style="whitegrid")
    
    # Choix d'une trajectoire représentative (la première)
    traj_idx = 0
    t0_truths = truths[traj_idx].cpu().numpy()
    t0_priors = priors[traj_idx].cpu().numpy()
    t0_knet = kalmannet_est[traj_idx].cpu().numpy()
    time_axis = np.arange(seq_len) * 0.01

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1])

    # --- PLOT 1 : Vue de dessus 2D (Trajectoire Globale) ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t0_truths[:, 0], t0_truths[:, 1], 'k-', linewidth=2, label="Vérité D1")
    ax1.plot(t0_priors[:, 0], t0_priors[:, 1], 'r--', alpha=0.6, label="EKF D1")
    ax1.plot(t0_knet[:, 0], t0_knet[:, 1], 'b-', linewidth=2, label="KalmanNet D1")
    
    ax1.plot(t0_truths[:, 2], t0_truths[:, 3], 'k:', linewidth=2, label="Vérité D2")
    ax1.plot(t0_priors[:, 2], t0_priors[:, 3], 'm--', alpha=0.6, label="EKF D2")
    ax1.plot(t0_knet[:, 2], t0_knet[:, 3], 'c-', linewidth=2, label="KalmanNet D2")
    
    ax1.set_title(f"Trajectoire 2D (Test ID : {traj_idx})", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Position X (m)")
    ax1.set_ylabel("Position Y (m)")
    ax1.legend(loc="upper right", ncol=2)
    ax1.axis('equal')

    # --- PLOT 2 & 3 : États Temporels (X et Y séparés pour D1) ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_axis, t0_truths[:, 0], 'k-', label="Vrai X")
    ax2.plot(time_axis, t0_priors[:, 0], 'r--', alpha=0.7, label="EKF X")
    ax2.plot(time_axis, t0_knet[:, 0], 'b-', alpha=0.8, label="KalmanNet X")
    ax2.set_title("Évolution de la variable d'état X_1(t)")
    ax2.set_ylabel("Position X (m)")
    
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax3.plot(time_axis, t0_truths[:, 1], 'k-', label="Vrai Y")
    ax3.plot(time_axis, t0_priors[:, 1], 'r--', alpha=0.7, label="EKF Y")
    ax3.plot(time_axis, t0_knet[:, 1], 'b-', alpha=0.8, label="KalmanNet Y")
    ax3.set_title("Évolution de la variable d'état Y_1(t)")
    ax3.set_ylabel("Position Y (m)")

    # --- PLOT 4 : Erreur Absolue dans le temps ---
    ax4 = fig.add_subplot(gs[2, 0], sharex=ax2)
    ax4.plot(time_axis, err_baseline_d1[traj_idx].cpu().numpy(), 'r--', label="Erreur EKF")
    ax4.plot(time_axis, err_knet_d1[traj_idx].cpu().numpy(), 'b-', label="Erreur KalmanNet")
    ax4.set_title("Erreur de Position D1 en fonction du temps")
    ax4.set_xlabel("Temps (s)")
    ax4.set_ylabel("Erreur Absolue (m)")
    ax4.legend()

    # --- PLOT 5 : Distribution des erreurs (Boxplot sur tout le dataset) ---
    # Cette visualisation est très prisée en Machine Learning pour montrer la robustesse
    ax5 = fig.add_subplot(gs[2, 1])
    # On aplatit les erreurs pour toutes les trajectoires et tous les pas de temps
    flat_err_b = err_baseline_d1.cpu().numpy().flatten()
    flat_err_k = err_knet_d1.cpu().numpy().flatten()
    
    error_data = pd.DataFrame({
        'Erreur (m)': np.concatenate([flat_err_b, flat_err_k]),
        'Modèle': ['EKF Baseline']*len(flat_err_b) + ['KalmanNet']*len(flat_err_k)
    })
    
    sns.boxplot(x='Modèle', y='Erreur (m)', data=error_data, ax=ax5, showfliers=False, palette=['#ff9999', '#99ccff'])
    ax5.set_title("Distribution des Erreurs (Vue sans valeurs extrêmes)")
    
    plt.tight_layout()
    plt.savefig("analyse_kalmannet_complete.png", dpi=300)
    print("✅ Dashboard complet sauvegardé sous 'analyse_kalmannet_complete.png'")
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()



import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

def generate_test_dataset():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Génération du Test Set sur : {device.type.upper()}")

    BATCH_SIZE = 500       # 500 trajectoires pour le test
    SIM_TIME = 20.0         
    DT_IMU = 0.01           
    DT_GPS = 1.0            
    DT_UWB = 0.1            

    STEPS_PER_TRAJ = int(SIM_TIME / DT_IMU)
    STEP_GPS = int(DT_GPS / DT_IMU)
    STEP_UWB = int(DT_UWB / DT_IMU)

    noise_scale_Q = torch.empty((BATCH_SIZE, 1, 1), device=device).uniform_(0.5, 2.0)
    noise_scale_R = torch.empty((BATCH_SIZE, 1, 1), device=device).uniform_(0.5, 2.0)

    Q_base_vals = [0.01, 0.01, 0.1, 0.1, np.deg2rad(1.0)]
    Q_single = torch.diag(torch.tensor(Q_base_vals, dtype=torch.float32, device=device)) ** 2
    Q_combined = torch.block_diag(Q_single, Q_single)
    Q_full = Q_combined.unsqueeze(0).repeat(BATCH_SIZE, 1, 1) * noise_scale_Q

    R_gps_base = torch.diag(torch.tensor([3.0, 3.0, 3.0, 3.0], dtype=torch.float32, device=device)) ** 2
    R_gps = R_gps_base.unsqueeze(0).repeat(BATCH_SIZE, 1, 1) * noise_scale_R

    R_uwb_base = torch.tensor([[[0.5 ** 2]]], dtype=torch.float32, device=device)
    R_uwb = R_uwb_base.repeat(BATCH_SIZE, 1, 1) * noise_scale_R

    X_true = torch.zeros((BATCH_SIZE, 10, 1), device=device)
    X_est = torch.zeros((BATCH_SIZE, 10, 1), device=device)

    start_y2 = torch.empty((BATCH_SIZE, 1, 1), device=device).uniform_(5.0, 15.0)
    X_true[:, 6:7, :] = start_y2
    X_est[:, 6:7, :] = start_y2

    P_est = torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) * 5.0
    bias_d2 = torch.empty((BATCH_SIZE, 3, 1), device=device).uniform_(-0.1, 0.1)

    H_gps = torch.zeros((BATCH_SIZE, 4, 10), device=device)
    H_gps[:, 0, 0] = 1; H_gps[:, 1, 1] = 1; H_gps[:, 2, 5] = 1; H_gps[:, 3, 6] = 1

    hist_features = torch.zeros((BATCH_SIZE, STEPS_PER_TRAJ, 25), dtype=torch.float32)

    current_ax = torch.zeros((BATCH_SIZE, 1), device=device)
    current_omega = torch.zeros((BATCH_SIZE, 1), device=device)

    print(f"⚙️ Simulation en cours...")
    for step in tqdm(range(STEPS_PER_TRAJ)):
        
        if step % 400 == 0:
            current_ax.uniform_(0.1, 1.0)
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
        u1 = u_base + torch.randn((BATCH_SIZE, 3, 1), device=device) * 0.05
        u2 = u_base + bias_d2 + torch.randn((BATCH_SIZE, 3, 1), device=device) * 0.05

        X_prev_est = X_est.clone()
        X_pred = torch.zeros_like(X_est)
        F = torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
        
        for idx, u in zip([0, 5], [u1, u2]):
            theta_e = X_est[:, idx+4, 0:1]
            a_x, a_y, om = u[:, 0, :], u[:, 1, :], u[:, 2, :]
            
            X_pred[:, idx, 0]   = X_est[:, idx, 0] + X_est[:, idx+2, 0] * DT_IMU
            X_pred[:, idx+1, 0] = X_est[:, idx+1, 0] + X_est[:, idx+3, 0] * DT_IMU
            X_pred[:, idx+2, 0] = X_est[:, idx+2, 0] + (a_x * torch.cos(theta_e) - a_y * torch.sin(theta_e)).squeeze(-1) * DT_IMU
            X_pred[:, idx+3, 0] = X_est[:, idx+3, 0] + (a_x * torch.sin(theta_e) + a_y * torch.cos(theta_e)).squeeze(-1) * DT_IMU
            X_pred[:, idx+4, 0] = theta_e.squeeze(-1) + om.squeeze(-1) * DT_IMU
            
            F[:, idx, idx+2] = DT_IMU; F[:, idx+1, idx+3] = DT_IMU
            F[:, idx+2, idx+4] = (-a_x * torch.sin(theta_e) - a_y * torch.cos(theta_e)).squeeze(-1) * DT_IMU
            F[:, idx+3, idx+4] = ( a_x * torch.cos(theta_e) - a_y * torch.sin(theta_e)).squeeze(-1) * DT_IMU
            
        P_pred = torch.bmm(F, torch.bmm(P_est, F.mT)) + Q_full
        X_est = X_pred.clone()
        delta_x = X_est - X_prev_est

        has_gps, has_uwb = 0.0, 0.0
        y_gps = torch.zeros((BATCH_SIZE, 4, 1), device=device)
        y_uwb = torch.zeros((BATCH_SIZE, 1, 1), device=device)
        
        if step % STEP_GPS == 0:
            has_gps = 1.0
            z_gps = torch.cat([X_true[:, 0:2, :], X_true[:, 5:7, :]], dim=1) + torch.randn((BATCH_SIZE, 4, 1), device=device) * 3.0 * torch.sqrt(noise_scale_R)
            y_gps = z_gps - torch.bmm(H_gps, X_est)
            
            S = torch.bmm(H_gps, torch.bmm(P_pred, H_gps.mT)) + R_gps
            K = torch.bmm(P_pred, torch.bmm(H_gps.mT, torch.linalg.inv(S)))
            X_est = X_est + torch.bmm(K, y_gps)
            P_pred = torch.bmm(torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) - torch.bmm(K, H_gps), P_pred)
            
        if step % STEP_UWB == 0:
            has_uwb = 1.0
            dist_t = torch.sqrt((X_true[:, 5, 0]-X_true[:, 0, 0])**2 + (X_true[:, 6, 0]-X_true[:, 1, 0])**2).unsqueeze(-1).unsqueeze(-1)
            z_dist = dist_t + torch.randn((BATCH_SIZE, 1, 1), device=device) * 0.5 * torch.sqrt(noise_scale_R)
            
            dx_e, dy_e = X_est[:, 5, 0]-X_est[:, 0, 0], X_est[:, 6, 0]-X_est[:, 1, 0]
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

    print("💾 Sauvegarde du Test Set...")
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
    df.to_pickle("data/kalman_dataset_test.pkl") # NOUVEAU NOM DE FICHIER
    print(f"✅ Test Set sauvegardé : data/kalman_dataset_test.pkl ({len(df)} lignes)")

if __name__ == "__main__":
    generate_test_dataset()


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. REDÉFINITION DE L'ARCHITECTURE
# ==========================================
class KalmanNet_Gain(nn.Module):
    def __init__(self, input_dim=17, obs_dim=5, state_dim=10, hidden_dim=64):
        super(KalmanNet_Gain, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.activation = nn.Tanh()
        self.fc = nn.Linear(hidden_dim, state_dim * obs_dim)

    def forward(self, features):
        x = features.transpose(1, 2)
        x = self.bn_input(x)
        x = x.transpose(1, 2)
        gru_out, _ = self.gru(x)
        gru_out = self.activation(gru_out)
        K_flat = self.fc(gru_out)
        K = K_flat.view(features.size(0), features.size(1), self.state_dim, self.obs_dim)
        return K

# ==========================================
# 2. FONCTION D'ÉVALUATION
# ==========================================
def evaluate_and_plot(model_path="weights/train4/kalmannet_best_weights.pth", test_data_path="data/kalman_dataset_test.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 Évaluation sur : {device.type.upper()}")

    # 1. Chargement du modèle
    model = KalmanNet_Gain().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Chargement et formatage des données de test
    print("Chargement du dataset de test...")
    df = pd.read_pickle(test_data_path)
    df = df.sort_values(by=['traj_id', 'time_step'])
    
    num_trajectories = df['traj_id'].nunique()
    seq_len = df['time_step'].nunique()
    
    data_np = df.drop(columns=['traj_id', 'time_step']).values
    data_tensor = torch.tensor(data_np, dtype=torch.float32).view(num_trajectories, seq_len, -1).to(device)
    
    features = data_tensor[:, :, 0:17] 
    y_t = data_tensor[:, :, 2:7].unsqueeze(-1) 
    priors = data_tensor[:, :, 17:21]   # Baseline EKF
    truths = data_tensor[:, :, 21:25]   # Vérité terrain

    # 3. Inférence massive (sans calcul de gradients)
    print("Inférence en cours...")
    with torch.no_grad():
        K = model(features)
        state_update = torch.matmul(K, y_t).squeeze(-1)
        pos_update = state_update[:, :, [0, 1, 5, 6]]
        kalmannet_est = priors + pos_update # Prédiction finale du KalmanNet

    # 4. Calcul des Métriques Globales (RMSE 2D)
    # Erreur Baseline = distance entre Prior et Vérité
    err_baseline_d1 = torch.sqrt((priors[:,:,0]-truths[:,:,0])**2 + (priors[:,:,1]-truths[:,:,1])**2)
    err_baseline_d2 = torch.sqrt((priors[:,:,2]-truths[:,:,2])**2 + (priors[:,:,3]-truths[:,:,3])**2)
    rmse_baseline = torch.mean((err_baseline_d1 + err_baseline_d2) / 2.0).item()

    # Erreur KalmanNet = distance entre KalmanNet et Vérité
    err_knet_d1 = torch.sqrt((kalmannet_est[:,:,0]-truths[:,:,0])**2 + (kalmannet_est[:,:,1]-truths[:,:,1])**2)
    err_knet_d2 = torch.sqrt((kalmannet_est[:,:,2]-truths[:,:,2])**2 + (kalmannet_est[:,:,3]-truths[:,:,3])**2)
    rmse_knet = torch.mean((err_knet_d1 + err_knet_d2) / 2.0).item()

    print("\n" + "="*40)
    print("🎯 RÉSULTATS GLOBAUX SUR LE TEST SET")
    print("="*40)
    print(f"RMSE Baseline (EKF) : {rmse_baseline:.3f} mètres")
    print(f"RMSE KalmanNet      : {rmse_knet:.3f} mètres")
    improvement = ((rmse_baseline - rmse_knet) / rmse_baseline) * 100
    print(f"Amélioration        : {improvement:.1f} %")
    print("="*40 + "\n")

    # 5. Visualisation de la Trajectoire N°0
    print("Génération du graphique...")
    t0_truths = truths[0].cpu().numpy()
    t0_priors = priors[0].cpu().numpy()
    t0_knet = kalmannet_est[0].cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Sous-graphe 1 : Trajectoires X/Y
    axs[0].plot(t0_truths[:, 0], t0_truths[:, 1], 'k-', linewidth=2, label="Vérité D1")
    axs[0].plot(t0_priors[:, 0], t0_priors[:, 1], 'r--', alpha=0.7, label="EKF Baseline D1")
    axs[0].plot(t0_knet[:, 0], t0_knet[:, 1], 'b-', linewidth=2, label="KalmanNet D1")
    
    axs[0].plot(t0_truths[:, 2], t0_truths[:, 3], 'k:', linewidth=2, label="Vérité D2")
    axs[0].plot(t0_priors[:, 2], t0_priors[:, 3], 'm--', alpha=0.7, label="EKF Baseline D2")
    axs[0].plot(t0_knet[:, 2], t0_knet[:, 3], 'c-', linewidth=2, label="KalmanNet D2")

    axs[0].set_title("Comparaison des Trajectoires (Test ID 0)")
    axs[0].set_xlabel("Position X (m)")
    axs[0].set_ylabel("Position Y (m)")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].axis('equal')

    # Sous-graphe 2 : Erreur Moyenne au cours du temps (pour D1)
    time_axis = np.arange(seq_len) * 0.01 # dt = 0.01
    err_b_np = err_baseline_d1[0].cpu().numpy()
    err_k_np = err_knet_d1[0].cpu().numpy()

    axs[1].plot(time_axis, err_b_np, 'r--', label="Erreur EKF Baseline")
    axs[1].plot(time_axis, err_k_np, 'b-', label="Erreur KalmanNet")
    axs[1].set_title("Évolution de l'Erreur de Position (Drone 1)")
    axs[1].set_xlabel("Temps (s)")
    axs[1].set_ylabel("Erreur (mètres)")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("test_comparison.png", dpi=300)
    print("✅ Graphique sauvegardé sous 'test_comparison.png'")
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()




import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importer ta classe KalmanNet_Gain depuis ton fichier d'entraînement
# from train_script import KalmanNet_Gain 

def run_test_inference(model_path, data_path, state_dim=10, obs_dim=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Charger le modèle
    model = KalmanNet_Gain(state_dim=state_dim, obs_dim=obs_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 2. Charger une trajectoire de test (ou générer via ton script de simulation)
    # Ici, on prend la première trajectoire d'un dataset de test
    df_test = pd.read_pickle(data_path)
    traj_id = df_test['traj_id'].unique()[0]
    df_single = df_test[df_test['traj_id'] == traj_id].sort_values('time_step')
    
    # Préparation des tenseurs
    features = torch.tensor(df_single.iloc[:, 2:19].values, dtype=torch.float32).unsqueeze(0).to(device)
    y_t = torch.tensor(df_single.iloc[:, 4:9].values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    priors = df_single[['prior_x1', 'prior_y1', 'prior_x2', 'prior_y2']].values
    truths = df_single[['true_x1', 'true_y1', 'true_x2', 'true_y2']].values
    
    # 3. Inférence
    with torch.no_grad():
        K = model(features)
        state_update = torch.matmul(K, y_t).squeeze(-1)
        pos_update = state_update[:, :, [0, 1, 5, 6]].cpu().numpy().squeeze(0)
        
    # 4. Reconstruction de la trajectoire estimée
    # Estimation = Prior (EKF prediction) + Correction (KalmanNet Gain * Innovation)
    estimations = priors + pos_update
    
    # 5. Visualisation
    plt.figure(figsize=(12, 6))
    plt.plot(truths[:, 0], truths[:, 1], 'k--', label="Vérité Terrain (Drone 1)")
    plt.plot(priors[:, 0], priors[:, 1], 'r:', label="Baseline EKF (Prior)")
    plt.plot(estimations[:, 0], estimations[:, 1], 'b-', label="KalmanNet")
    plt.title(f"Comparaison sur trajectoire de test #{traj_id}")
    plt.legend()
    plt.grid(True)
    plt.show()

# run_test_inference("weights/train4/kalmannet_best_weights.pth", "data/test_dataset.pkl")



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os

# ==========================================
# 1. PARAMÈTRES ET CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Appareil utilisé : {device.type.upper()}")

# Création du dossier de poids si inexistant
os.makedirs("weights/train4", exist_ok=True)

# ==========================================
# 2. ARCHITECTURE DU KALMANNET OPTIMISÉE
# ==========================================
class KalmanNet_Gain(nn.Module):
    def __init__(self, input_dim=17, obs_dim=5, state_dim=10, hidden_dim=64):
        super(KalmanNet_Gain, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # Normalisation des entrées pour stabiliser le GRU
        self.bn_input = nn.BatchNorm1d(input_dim)
        
        # GRU multicouche
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        
        # Activation Tanh pour borner l'état caché avant la FC
        self.activation = nn.Tanh()
        
        # Couche de sortie
        self.fc = nn.Linear(hidden_dim, state_dim * obs_dim)
        
        # Initialisation très faible pour commencer avec un Gain quasi-nul (confiance EKF)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, features):
        # features: [B, Seq, 17]
        # BatchNorm attend [B, C, L], donc on transpose
        x = features.transpose(1, 2)
        x = self.bn_input(x)
        x = x.transpose(1, 2)
        
        gru_out, _ = self.gru(x)
        gru_out = self.activation(gru_out)
        
        K_flat = self.fc(gru_out)
        
        # Reshape en [Batch, Seq, 10, 5]
        K = K_flat.view(features.size(0), features.size(1), self.state_dim, self.obs_dim)
        return K

# ==========================================
# 3. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==========================================
def load_and_prepare_data(pickle_path="data/kalman_dataset_gpu_.pkl", batch_size=256):
    print("💾 Chargement des données...")
    df = pd.read_pickle(pickle_path)
    
    df = df.sort_values(by=['traj_id', 'time_step'])
    num_trajectories = df['traj_id'].nunique()
    seq_len = df['time_step'].nunique()
    
    data_np = df.drop(columns=['traj_id', 'time_step']).values
    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    data_tensor = data_tensor.view(num_trajectories, seq_len, -1)
    
    # Slicing des colonnes selon ton format (0:25)
    features = data_tensor[:, :, 0:17] 
    y_t = data_tensor[:, :, 2:7].unsqueeze(-1) 
    priors = data_tensor[:, :, 17:21]
    targets = data_tensor[:, :, 21:25]

    dataset = TensorDataset(features, y_t, priors, targets)
    
    train_size = int(0.8 * num_trajectories)
    val_size = num_trajectories - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# ==========================================
# 4. BOUCLE D'ENTRAÎNEMENT AMÉLIORÉE
# ==========================================
def train_model():
    train_loader, val_loader = load_and_prepare_data()
    
    model = KalmanNet_Gain().to(device)
    
    # Optimizer avec Weight Decay pour éviter l'overfitting
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Scheduler : Réduit le LR si la val_loss ne s'améliore pas pendant 10 époques
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    criterion = nn.SmoothL1Loss()
    
    epochs = 300
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    print(f"\n🔥 Début de l'entraînement pour {epochs} époques...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_yt, batch_priors, batch_targets in train_loader:
            batch_features, batch_yt = batch_features.to(device), batch_yt.to(device)
            batch_priors, batch_targets = batch_priors.to(device), batch_targets.to(device)
            
            optimizer.zero_grad()
            
            # 1. Prédire K
            K = model(batch_features)
            
            # 2. Update : x_est = x_prior + K * y
            # K: [B, S, 10, 5], yt: [B, S, 5, 1] -> [B, S, 10, 1]
            state_update = torch.matmul(K, batch_yt).squeeze(-1) 
            
            # On ne corrige que les positions [x1, y1, x2, y2]
            pos_update = state_update[:, :, [0, 1, 5, 6]]
            estimated_positions = batch_priors + pos_update
            
            loss = criterion(estimated_positions, batch_targets)
            loss.backward()
            
            # Gradient clipping pour la stabilité des RNN
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
            
        # Phase de validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_yt, batch_priors, batch_targets in val_loader:
                batch_features, batch_yt = batch_features.to(device), batch_yt.to(device)
                batch_priors, batch_targets = batch_priors.to(device), batch_targets.to(device)
                
                K = model(batch_features)
                state_update = torch.matmul(K, batch_yt).squeeze(-1)
                pos_update = state_update[:, :, [0, 1, 5, 6]]
                estimated_positions = batch_priors + pos_update
                
                loss = criterion(estimated_positions, batch_targets)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Mise à jour du scheduler
        scheduler.step(avg_val_loss)
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Sauvegarde du meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "weights/train4/kalmannet_best_weights.pth")
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6e}")

    # Finalisation
    pd.DataFrame(history).to_csv("training_history.csv", index=False)
    torch.save(model.state_dict(), "weights/train4/kalmannet_final_weights.pth")
    print("\n✅ Entraînement terminé !")

if __name__ == "__main__":
    train_model()


mport torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

# ==========================================
# 1. PARAMÈTRES ET CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device.type.upper()}")

# ==========================================
# 2. ARCHITECTURE DU KALMANNET (SEQ2SEQ)
# ==========================================
class KalmanNet_Gain(nn.Module):
    def __init__(self, input_dim=17, obs_dim=5, state_dim=10, hidden_dim=64):
        """
        Le réseau calcule uniquement le Gain K.
        - input_dim (17) : has_gps(1) + has_uwb(1) + y_gps(4) + y_uwb(1) + dx(10)
        - obs_dim (5) : Les 5 innovations (4 GPS + 1 UWB)
        - state_dim (10) : Les 10 dimensions de ton système pour la matrice K
        """
        super(KalmanNet_Gain, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # Un GRU pour modéliser l'évolution de l'incertitude dans le temps
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.activation = nn.Tanh() #Pour stabliliser les caractéristiques du GRU
        # Couche de sortie pour prédire les 50 valeurs (10x5) de la matrice de Gain
        self.fc = nn.Linear(hidden_dim, state_dim * obs_dim)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01) #On setup les paramètres de la couche à 0 pour pour que la couche fasse confiance au début au modèle et moins au gps
        nn.init.constant_(self.fc.biais, 0)

    def forward(self, features):
        # features shape: [batch_size, seq_len, 17]
        gru_out, _ = self.gru(features)
        gru_out = self.activation(gru_out)
        
        # On passe la séquence entière dans le réseau dense
        K_flat = self.fc(gru_out)
        
        # On reshape pour avoir une séquence de matrices K
        # K shape : [batch_size, seq_len, 10, 5]
        K = K_flat.view(features.size(0), features.size(1), self.state_dim, self.obs_dim)
        return K

# ==========================================
# 3. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==========================================
def load_and_prepare_data(pickle_path="../dataset/data/kalman_dataset_gpu_.pkl", batch_size=256):
    print("Chargement des données...")
    df = pd.read_pickle(pickle_path)
    
    # On vérifie les dimensions déduites du dataset
    num_trajectories = df['traj_id'].nunique()
    seq_len = df['time_step'].nunique()
    
    # Tri sécurisé pour garantir l'ordre temporel
    df = df.sort_values(by=['traj_id', 'time_step'])
    
    # Extraction des tenseurs en numpy puis torch
    data_np = df.drop(columns=['traj_id', 'time_step']).values
    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    
    # Reshape en [Batch, Sequence, Features]
    data_tensor = data_tensor.view(num_trajectories, seq_len, -1)
    
    # --- Indexation basée sur tes colonnes ---
    # Colonnes restantes (0 à 24) :
    # 0: has_gps | 1: has_uwb | 2:5: y_gps (4) | 6: y_uwb (1) | 7:16: dx (10)
    # 17:20: prior (4) | 21:24: true (4)
    
    # Features pour le réseau (17 dimensions : has_*, innovations, dx)
    features = data_tensor[:, :, 0:17] 
    
    # Vecteur d'innovation y_t (5 dimensions : 4 GPS + 1 UWB)
    y_t = data_tensor[:, :, 2:7].unsqueeze(-1) # [B, Seq, 5, 1] pour la multiplication matricielle
    
    # États a priori (prédictions EKF) pour l'addition finale (4 positions)
    priors = data_tensor[:, :, 17:21]
    
    # Vérité terrain pour le calcul de la loss (4 positions)
    targets = data_tensor[:, :, 21:25]

    dataset = TensorDataset(features, y_t, priors, targets)
    
    # Division Train / Validation (80% / 20%)
    train_size = int(0.8 * num_trajectories)
    val_size = num_trajectories - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# ==========================================
# 4. BOUCLE D'ENTRAÎNEMENT
# ==========================================
def train_model():
    train_loader, val_loader = load_and_prepare_data()
    
    model = KalmanNet_Gain().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss()
    
    epochs = 300
    print("\n Début de l'entraînement...")
    
    # --- NOUVEAU : Dictionnaire pour l'historique ---
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}
    
    # Pour sauvegarder le meilleur modèle
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # --- Phase d'entraînement ---
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_yt, batch_priors, batch_targets in train_loader:
            batch_features, batch_yt = batch_features.to(device), batch_yt.to(device)
            batch_priors, batch_targets = batch_priors.to(device), batch_targets.to(device)
            
            optimizer.zero_grad()
            
            K = model(batch_features)
            state_update = torch.matmul(K, batch_yt).squeeze(-1) 
            pos_update = state_update[:, :, [0, 1, 5, 6]]
            estimated_positions = batch_priors + pos_update
            
            loss = criterion(estimated_positions, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
            
        # --- Phase de validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_yt, batch_priors, batch_targets in val_loader:
                batch_features, batch_yt = batch_features.to(device), batch_yt.to(device)
                batch_priors, batch_targets = batch_priors.to(device), batch_targets.to(device)
                
                K = model(batch_features)
                state_update = torch.matmul(K, batch_yt).squeeze(-1)
                pos_update = state_update[:, :, [0, 1, 5, 6]]
                estimated_positions = batch_priors + pos_update
                
                loss = criterion(estimated_positions, batch_targets)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Enregistrement des métriques ---
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Sauvegarde conditionnelle du meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "weights/train4/kalmannet_best_weights.pth")
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Sauvegarde de l'historique et du modèle final ---
    df_history = pd.DataFrame(history)
    df_history.to_csv("training_history.csv", index=False)
    torch.save(model.state_dict(), "weights/train4/kalmannet_final_weights.pth")
    
    print("\n Entraînement terminé !")
    print(" Historique sauvegardé sous 'training_history.csv'")
    print(" Meilleur modèle sauvegardé sous 'kalmannet_best_weights.pth'")


if __name__ == "__main__":
    train_model()

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

BATCH_SIZE = 5000       # Nombre de trajectoires simultanées
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
# Facteurs d'échelle aléatoires pour chaque trajectoire du batch
noise_scale_Q = torch.empty((BATCH_SIZE, 1, 1), device=device).uniform_(0.5, 2.0)
noise_scale_R = torch.empty((BATCH_SIZE, 1, 1), device=device).uniform_(0.5, 2.0)

# Construction de Q_full (10x10) pour 2 drones
Q_base_vals = [0.01, 0.01, 0.1, 0.1, np.deg2rad(1.0)]
Q_single = torch.diag(torch.tensor(Q_base_vals, dtype=torch.float32, device=device)) ** 2
Q_combined = torch.block_diag(Q_single, Q_single) # On assemble Drone 1 et Drone 2
Q_full = Q_combined.unsqueeze(0).repeat(BATCH_SIZE, 1, 1) * noise_scale_Q

# Construction de R_gps (4x4 : x1, y1, x2, y2)
R_gps_base = torch.diag(torch.tensor([3.0, 3.0, 3.0, 3.0], dtype=torch.float32, device=device)) ** 2
R_gps = R_gps_base.unsqueeze(0).repeat(BATCH_SIZE, 1, 1) * noise_scale_R

# Construction de R_uwb (1x1)
R_uwb_base = torch.tensor([[[0.5 ** 2]]], dtype=torch.float32, device=device)
R_uwb = R_uwb_base.repeat(BATCH_SIZE, 1, 1) * noise_scale_R

# États (Vrai et Estimé)
X_true = torch.zeros((BATCH_SIZE, 10, 1), device=device)
X_est = torch.zeros((BATCH_SIZE, 10, 1), device=device)

# Position Y initiale aléatoire pour séparer les drones
start_y2 = torch.empty((BATCH_SIZE, 1, 1), device=device).uniform_(5.0, 15.0)
X_true[:, 6:7, :] = start_y2
X_est[:, 6:7, :] = start_y2

# Covariance initiale
P_est = torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) * 5.0

# Biais IMU constant par trajectoire pour le Drone 2
bias_d2 = torch.empty((BATCH_SIZE, 3, 1), device=device).uniform_(-0.1, 0.1)

# Matrice H GPS (Fixe : extrait x1, y1, x2, y2)
H_gps = torch.zeros((BATCH_SIZE, 4, 10), device=device)
H_gps[:, 0, 0] = 1; H_gps[:, 1, 1] = 1; H_gps[:, 2, 5] = 1; H_gps[:, 3, 6] = 1

# Historique de stockage (25 features : flags, innovations, dx, priors, truths)
hist_features = torch.zeros((BATCH_SIZE, STEPS_PER_TRAJ, 25), dtype=torch.float32)

current_ax = torch.zeros((BATCH_SIZE, 1), device=device)
current_omega = torch.zeros((BATCH_SIZE, 1), device=device)

# ==========================================
# 3. LA BOUCLE TEMPORELLE BATCHÉE
# ==========================================
print(f"⚙️ Génération du dataset ({BATCH_SIZE} trajectoires)...")

for step in tqdm(range(STEPS_PER_TRAJ)):
    
    # Changement de consigne de vol toutes les 4 secondes
    if step % 400 == 0:
        current_ax.uniform_(0.1, 1.0)
        current_omega.uniform_(-0.5, 0.5)
        current_ay = torch.zeros_like(current_ax)
        
    # --- 1. ÉVOLUTION DE LA VÉRITÉ (f non-linéaire) ---
    for i in [0, 5]: 
        theta_t = X_true[:, i+4, 0:1]
        X_true[:, i, 0:1]   += X_true[:, i+2, 0:1] * DT_IMU
        X_true[:, i+1, 0:1] += X_true[:, i+3, 0:1] * DT_IMU
        X_true[:, i+2, 0:1] += (current_ax * torch.cos(theta_t) - current_ay * torch.sin(theta_t)) * DT_IMU
        X_true[:, i+3, 0:1] += (current_ax * torch.sin(theta_t) + current_ay * torch.cos(theta_t)) * DT_IMU
        X_true[:, i+4, 0:1] += current_omega * DT_IMU
        
    # --- 2. COMMANDES IMU BRUITÉES ---
    u_base = torch.cat([current_ax.unsqueeze(-1), current_ay.unsqueeze(-1), current_omega.unsqueeze(-1)], dim=1) 
    u1 = u_base + torch.randn((BATCH_SIZE, 3, 1), device=device) * 0.05
    u2 = u_base + bias_d2 + torch.randn((BATCH_SIZE, 3, 1), device=device) * 0.05

    # --- 3. PRÉDICTION EKF (Linéarisation F) ---
    X_prev_est = X_est.clone()
    X_pred = torch.zeros_like(X_est)
    F = torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
    
    for idx, u in zip([0, 5], [u1, u2]):
        theta_e = X_est[:, idx+4, 0:1]
        a_x, a_y, om = u[:, 0, :], u[:, 1, :], u[:, 2, :]
        
        # Modèle de transition f(x, u)
        X_pred[:, idx, 0]   = X_est[:, idx, 0] + X_est[:, idx+2, 0] * DT_IMU
        X_pred[:, idx+1, 0] = X_est[:, idx+1, 0] + X_est[:, idx+3, 0] * DT_IMU
        X_pred[:, idx+2, 0] = X_est[:, idx+2, 0] + (a_x * torch.cos(theta_e) - a_y * torch.sin(theta_e)).squeeze(-1) * DT_IMU
        X_pred[:, idx+3, 0] = X_est[:, idx+3, 0] + (a_x * torch.sin(theta_e) + a_y * torch.cos(theta_e)).squeeze(-1) * DT_IMU
        X_pred[:, idx+4, 0] = theta_e.squeeze(-1) + om.squeeze(-1) * DT_IMU
        
        # Calcul de la Jacobienne F
        F[:, idx, idx+2] = DT_IMU
        F[:, idx+1, idx+3] = DT_IMU
        F[:, idx+2, idx+4] = (-a_x * torch.sin(theta_e) - a_y * torch.cos(theta_e)).squeeze(-1) * DT_IMU
        F[:, idx+3, idx+4] = ( a_x * torch.cos(theta_e) - a_y * torch.sin(theta_e)).squeeze(-1) * DT_IMU
        
    # Propagation de la covariance et de l'état
    P_pred = torch.bmm(F, torch.bmm(P_est, F.mT)) + Q_full
    X_est = X_pred.clone()
    delta_x = X_est - X_prev_est

    has_gps, has_uwb = 0.0, 0.0
    y_gps = torch.zeros((BATCH_SIZE, 4, 1), device=device)
    y_uwb = torch.zeros((BATCH_SIZE, 1, 1), device=device)
    
    # --- 4. UPDATE GPS ---
    if step % STEP_GPS == 0:
        has_gps = 1.0
        z_gps = torch.cat([X_true[:, 0:2, :], X_true[:, 5:7, :]], dim=1) + torch.randn((BATCH_SIZE, 4, 1), device=device) * 3.0 * torch.sqrt(noise_scale_R)
        y_gps = z_gps - torch.bmm(H_gps, X_est)
        
        S = torch.bmm(H_gps, torch.bmm(P_pred, H_gps.mT)) + R_gps
        K = torch.bmm(P_pred, torch.bmm(H_gps.mT, torch.linalg.inv(S)))
        X_est = X_est + torch.bmm(K, y_gps)
        P_pred = torch.bmm(torch.eye(10, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) - torch.bmm(K, H_gps), P_pred)
        
    # --- 5. UPDATE UWB ---
    if step % STEP_UWB == 0:
        has_uwb = 1.0
        dist_t = torch.sqrt((X_true[:, 5, 0]-X_true[:, 0, 0])**2 + (X_true[:, 6, 0]-X_true[:, 1, 0])**2).unsqueeze(-1).unsqueeze(-1)
        z_dist = dist_t + torch.randn((BATCH_SIZE, 1, 1), device=device) * 0.5 * torch.sqrt(noise_scale_R)
        
        dx_e, dy_e = X_est[:, 5, 0]-X_est[:, 0, 0], X_est[:, 6, 0]-X_est[:, 1, 0]
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

    # --- 6. STOCKAGE ---
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
# 4. EXPORT PANDAS / PICKLE
# ==========================================
print("💾 Fin de simulation. Formatage en DataFrame...")
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
print(f"✅ Dataset sauvegardé : data/kalman_dataset_gpu_.pkl ({len(df)} lignes)")

    
    
    
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
