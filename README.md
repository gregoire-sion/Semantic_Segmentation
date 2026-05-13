import torch 
import torch.nn as nn 
from .base_filter import BaseFilter 
from ..config import Config as config

def batch_rotation_matrix(angles_norm, scale_angle=torch.pi):
    #Dénormalisation dans un premier temps
    angles = angles_norm *scale_angle
    phi = angles[..., 0]
    theta = angles[..., 1]
    psi = angles[..., 2]

    c_phi, s_phi = torch.cos(phi), torch.sin(phi)
    c_theta, s_theta = torch.cos(theta), torch.sin(theta)
    c_psi, s_psi = torch.cos(psi), torch.sin(psi)

    R = torch.zeros((*phi.shape, 3, 3), device=phi.device)

    R[..., 0, 0] = c_theta * c_psi
    R[..., 0, 1] = s_phi * s_theta * c_psi - c_phi * s_psi
    R[..., 0, 2] = c_phi * s_theta * c_psi + s_phi * s_psi

    R[..., 1, 0] = c_theta * s_psi
    R[..., 1, 1] = s_phi * s_theta * s_psi + c_phi * c_psi
    R[..., 1, 2] = c_phi * s_theta * s_psi - s_phi * c_psi

    R[..., 2, 0] = -s_theta
    R[..., 2, 1] = s_phi * c_theta
    R[..., 2, 2] =c_phi* c_theta
    
    return R


class KalmanNet(BaseFilter):
    "KalmanNet"

    def __init__(self, config):
        super().__init__(state_dim=config.state_dim, obs_dim=config.obs_dim, device=config.device)
        self.config = config
        self.state_dim = config.state_dim
        self.obs_dim = config.obs_dim
        self.device = config.device
        self.n_drones = config.n_drones
        self.hidden_dim = config.hidden_dim
        
        input_dim = 9 + (self.n_drones - 1)

        H = torch.zeros((self.obs_dim,self.state_dim))
        H[0,0] = 1.0
        H[1,2] = 1.0

        self.register_buffer('H', H)
        self.register_buffer('I', torch.eye(self.state_dim))

        self.rnn = nn.GRU(
            input_size=self.obs_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=2, 
            dropout=0.0,
            batch_first=True
            )

        self.fc = nn.Linear(
            self.hidden_dim, 
            self.state_dim * self.obs_dim
            )

        self.apply(self.init_weights)

        if config.IS_NORMALISATION :
            self.ratio_v_x = config.scale_vel / config.scale_pos
            self.ratio_a_v = config.scale_acc / config.scale_vel
            self.ratio_g_a = config.scale_gyro / torch.pi
        else :
            self.ratio_v_x = 1.0
            self.ratio_a_v = 1.0
            self.ratio_g_a = 1.0
    
    def forward(self, batch_init, batch_data):
        current_batch_size = batch_data.shape[0]
        n_drones = self.n_drones
        seq_len = batch_data.shape[1]

        # x_est shape: (Batch, N_drones, 9)
        # Index: 0:3 (Pos), 3:6 (Vel), 6:9 (Angles)
        x_est = batch_init.clone()
        h_rnn = None # État caché du GRU 

        estimations = []
        
        dt = self.config.dt
        dt_imu = self.config.dt_imu
        dt_gps = self.config.dt_gps

        # Dimensions pour le redimensionnement du RNN
        state_dim = config.state_dim
        obs_dim = config.obs_dim

        for t in range(seq_len):

            # ---------------------------------------------------------
            # 1. LECTURE DE L'IMU (Si la fréquence le permet)
            # ---------------------------------------------------------
            if t % dt_imu == 0: 
                # On lit les données pour tous les drones d'un coup
                # Index basé sur notre Dataset : 9:12 (Accel), 12:15 (Gyro)
                a_body = batch_data[:, t, :, 9:12]
                gyro = batch_data[:, t, :, 12:15]

            # ---------------------------------------------------------
            # 2. ÉTAPE DE PROPAGATION (Physique 3D)
            # ---------------------------------------------------------
            pos_est = x_est[:, :, 0:3]
            vel_est = x_est[:, :, 3:6]
            angles_est = x_est[:, :, 6:9]

            # A. Intégration des angles
            angles_pred = angles_est + gyro * dt * self.ratio_g_a

            # B. Projection de l'accélération (Drone -> Monde)
            R = batch_rotation_matrix(angles_pred) # Shape: (B, N, 3, 3)
            
            a_body_real = a_body * self.config.scale_acc
            a_body_real = a_body_real.unsqueeze(-1)
            
            a_earth_real = torch.matmul(R, a_body_real).squeeze(-1)
            a_earth_real[..., 2] -= 9.81 # Retrait de la gravité sur l'axe Z
            
            a_earth_norm = a_earth_real / self.config.scale_acc

            # C. Intégration de la cinématique
            vel_pred = vel_est + a_earth_norm * dt * self.ratio_a_v
            pos_pred = pos_est + vel_pred * dt * self.ratio_v_x

            angles_pred = torch.clamp(angles_pred, min=-10.0, max=10.0)
            vel_pred = torch.clamp(vel_pred, min=-50.0, max=50.0)
            pos_pred = torch.clamp(pos_pred, min=-5000.0, max=5000.0)

            # Recomposition du vecteur d'état prédit
            x_pred = torch.cat([pos_pred, vel_pred, angles_pred], dim=-1)
        
            # ---------------------------------------------------------
            # 3. ÉTAPE DU CALCUL DU GAIN & CORRECTION
            # ---------------------------------------------------------
            if t % dt_gps == 0: 
                # --- A. Calcul de l'innovation GPS (y_gps) ---
                gps_mesure = batch_data[:, t, :, 15:18]
                y_gps = gps_mesure - pos_pred

                temps_t = batch_data[:, t, :, 18]

                # --- B. Calcul de l'innovation de Distance (y_dist) ---
                dist_mesure = batch_data[:, t, :, 19:] # Toutes les distances mesurées
                dist_preds = []
                
                # On calcule la distance prédite entre chaque paire de drones
                for d in range(n_drones):
                    d_neighbors_dist = []
                    for autre_d in range(n_drones):
                        if d != autre_d:
                            # Distance Euclidienne L2
                            diff = pos_pred[:, d, :] - pos_pred[:, autre_d, :]
                            dist = torch.norm(diff + 1e-8, dim=-1)
                            d_neighbors_dist.append(dist)

                    dist_preds.append(torch.stack(d_neighbors_dist, dim=1))
                
                y_dist_pred = torch.stack(dist_preds, dim=1) # Shape: (Batch, N, N-1)
                y_dist = dist_mesure - y_dist_pred

                # --- C. Le vecteur d'innovation global (y) ---
                y = torch.cat([y_gps, y_dist], dim=-1) # Shape: (Batch, N, obs_dim)

                y = torch.clamp(y, min=-50.0, max=50.0)
                # --- D. Le RNN choisit le gain de Kalman ---
                # On "aplatit" Batch et Drones pour traiter tout le monde en parallèle
                y_flat = y.view(current_batch_size * self.n_drones, self.obs_dim)
                x_pred_flat = x_pred.view(current_batch_size * self.n_drones, self.state_dim)

                out, h_rnn = self.rnn(y_flat.unsqueeze(1), h_rnn)
                k_flat = self.fc(out.squeeze(1)) # self.fc doit avoir une sortie de taille (state_dim * obs_dim)
                
                # On remet le gain sous forme de Matrice (Batch*N, State_dim, Obs_dim)
                K = k_flat.view(current_batch_size * n_drones, state_dim, obs_dim)
                K = torch.sigmoid(K) # ou torch.tanh si tu veux autoriser des gains négatifs

                # --- E. Mise à jour (Correction de l'état) ---
                correction = torch.bmm(K, y_flat.unsqueeze(-1)).squeeze(-1)
                x_est_flat = x_pred_flat + correction

                x_est_flat = torch.clamp(x_est_flat, min=-5000.0, max=5000.0)
                
                # On redonne sa forme 3D à l'état
                x_est = x_est_flat.view(current_batch_size, self.n_drones, self.state_dim)

            else: 
                x_est = x_pred

            
            # # --- Sécurité Anti-Explosion ---
            # if x_est.abs().max() > 10000: 
            #     print(f" EXPLOSION DÉTECTÉE au pas de temps t={t} !")
            #     print(f" - Valeur max de x_est : {x_est.abs().max().item():.2e}")
            #     break

            if torch.isnan(x_est).any():
                print(f"Le premier Nan est apparu au telos t = {t}")

                if torch.isnan(pos_pred).any : print("Coupable pos pred (physique/viresse)")
                if torch.isnan(angles_pred).any(): print("Coupable angles_pred (IMU Rotation)")
                if t%dt_gps==0 and torch.isnan(K).any(): print("Coupable matrice K (Reseau de neurones / sigmoid)")
                raise ValueError("Arrer d'urgence pour Nan")

                max_val = x_est.abs().max().item()
                if max_val > 10000:
                    print(f"Attention valeurs gigantestques detectées a t = {t} ( Max: {max_val:.2e})")

            estimations.append(x_est.clone())


        # On retourne l'historique complet des états
        return torch.stack(estimations, dim=1)



    def init_weights(self, m) : 
        if isinstance(m, nn.Linear) :
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None : 
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM) :
            for name, param in m.named_parameters() :
                if 'weigth_ih' in name : 
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name : 
                    nn.init.orthogonal_(param)
                elif 'bias' in name : 
                    nn.init.constant_(param, 0)
