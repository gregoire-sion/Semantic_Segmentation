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
