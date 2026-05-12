import torch
import torch.nn as nn

# (Assure-toi d'avoir la fonction batch_rotation_matrix définie au-dessus de ta classe, 
# comme je te l'ai donnée dans le message précédent)

    def forward(self, batch_init, batch_data):
        current_batch_size = batch_data.shape[0]
        n_drones = self.n_drones
        seq_len = self.config.seq_len

        # x_est shape: (Batch, N_drones, 9)
        # Index: 0:3 (Pos), 3:6 (Vel), 6:9 (Angles)
        x_est = batch_init.clone()
        h_rnn = None # État caché du GRU 

        estimations = []
        
        dt = self.config.dt
        dt_imu = self.config.dt_imu
        dt_gps = self.config.dt_gps

        # Dimensions pour le redimensionnement du RNN
        state_dim = 9
        obs_dim = 3 + (n_drones - 1) # GPS 3D + Distances

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

            # Recomposition du vecteur d'état prédit
            x_pred = torch.cat([pos_pred, vel_pred, angles_pred], dim=-1)

            # ---------------------------------------------------------
            # 3. ÉTAPE DU CALCUL DU GAIN & CORRECTION
            # ---------------------------------------------------------
            if t % dt_gps == 0: 
                # --- A. Calcul de l'innovation GPS (y_gps) ---
                gps_mesure = batch_data[:, t, :, 15:18]
                y_gps = gps_mesure - pos_pred

                # --- B. Calcul de l'innovation de Distance (y_dist) ---
                dist_mesure = batch_data[:, t, :, 18:] # Toutes les distances mesurées
                dist_preds = []
                
                # On calcule la distance prédite entre chaque paire de drones
                for d in range(n_drones):
                    d_dists = []
                    for autre_d in range(n_drones):
                        if d != autre_d:
                            # Distance Euclidienne L2
                            dist = torch.norm(pos_pred[:, d, :] - pos_pred[:, autre_d, :], dim=-1)
                            d_dists.append(dist)
                    dist_preds.append(torch.stack(d_dists, dim=1))
                
                y_dist_pred = torch.stack(dist_preds, dim=1) # Shape: (Batch, N, N-1)
                y_dist = dist_mesure - y_dist_pred

                # --- C. Le vecteur d'innovation global (y) ---
                y = torch.cat([y_gps, y_dist], dim=-1) # Shape: (Batch, N, obs_dim)

                # --- D. Le RNN choisit le gain de Kalman ---
                # On "aplatit" Batch et Drones pour traiter tout le monde en parallèle
                y_flat = y.view(current_batch_size * n_drones, obs_dim)
                x_pred_flat = x_pred.view(current_batch_size * n_drones, state_dim)

                out, h_rnn = self.rnn(y_flat.unsqueeze(1), h_rnn)
                k_flat = self.fc(out.squeeze(1)) # self.fc doit avoir une sortie de taille (state_dim * obs_dim)
                
                # On remet le gain sous forme de Matrice (Batch*N, State_dim, Obs_dim)
                K = k_flat.view(current_batch_size * n_drones, state_dim, obs_dim)
                K = torch.sigmoid(K) # ou torch.tanh si tu veux autoriser des gains négatifs

                # --- E. Mise à jour (Correction de l'état) ---
                # x_est = x_pred + K * y
                x_est_flat = x_pred_flat + torch.bmm(K, y_flat.unsqueeze(-1)).squeeze(-1)
                
                # On redonne sa forme 3D à l'état
                x_est = x_est_flat.view(current_batch_size, n_drones, state_dim)

            else: 
                x_est = x_pred

            estimations.append(x_est.clone())

            # # --- Sécurité Anti-Explosion ---
            # if x_est.abs().max() > 10000: 
            #     print(f"💥 EXPLOSION DÉTECTÉE au pas de temps t={t} !")
            #     print(f" - Valeur max de x_est : {x_est.abs().max().item():.2e}")
            #     break

        # On retourne l'historique complet des états
        return torch.stack(estimations, dim=1)

  
  
  
  def forward(self, batch_init, batch_data) :

        current_batch_size = batch_data.shape[0]
        batch_size = self.config.batch_size
        seq_len = self.config.seq_len

        x_est = batch_init.clone()
        h_rnn = None #Etat caché du GRU 

        estimations = []
        covariances = []

        dt = self.config.dt
        dt_imu = self.config.dt_imu
        dt_gps = self.config.dt_gps

        for t in range (seq_len):

            if t % dt_imu == 0 : # Dans le cas ou l'imu communique
                ax = batch_data[:,t,5]
                ay = batch_data[:,t,6]
                omega = batch_data[:,t,7]

            #--------------------
            #ETAPE DE PROPAGATION
            #--------------------
            theta = x_est[:,4]
            x_pred = x_est.clone()
            x_pred[:,0] = x_est[:,0] + x_est[:,1]*dt*self.ratio_v_x
            x_pred[:,1] = x_est[:,1] + (ax*torch.cos(theta) - ay*torch.sin(theta))*dt*self.ratio_a_v
            x_pred[:,2] = x_est[:,2] + x_est[:,3]*dt*self.ratio_v_x
            x_pred[:,3] = x_est[:,3] + (ax*torch.sin(theta) + ay*torch.cos(theta))*dt*self.ratio_a_v
            x_pred[:,4] = x_est[:,4] + omega*dt*self.ratio_o_t
            

            #-----------------------
            #ETAPE DU CALCUL DU GAIN
            #-----------------------

            if t % dt_gps == 0 : 
                y = batch_data[:,t,11:13] - torch.matmul(x_pred,self.H.t())

                #-----Le RNN choisi le gain de Kalman----
                out, h_rnn = self.rnn(y.unsqueeze(1), h_rnn)
                k_flat = self.fc(out.squeeze(1))
                K = k_flat.view(current_batch_size, self.state_dim, self.obs_dim)
                K= torch.sigmoid(K)

                #-------Mise à jour ----------
                x_est = x_pred + torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)
            else : 
                x_est = x_pred

            x_est_denorm = x_est * self.config.scale_pos
            estimations.append(x_est.clone())

            # if x_est.abs().max() > 10000 : 
            #     print(f"EXPLOSION DETECTEE au pas de temps t={t} !")
            #     print(f" - Valeur max de x_est : {x_est.abs().max().item():.2e}")
            #     print(f" - Gain K : {K.abs().max().item():.2e}")
            #     break

        return torch.stack(estimations, dim=1) 
