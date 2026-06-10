
import numpy as np
from scipy.optimize import minimize_scalar

class DistributedDrone:
    def __init__(self, drone_id, x_init, P_init, Q_local):
        """
        Chaque drone gère uniquement son propre état (dim 8) et sa covariance (8x8)
        """
        self.id = drone_id
        self.x_est = x_init.copy()  # [x, y, vx, vy, ax, ay, bx, by]
        self.P_est = P_init.copy()
        self.Q = Q_local.copy()
        
        # Matrice de commande locale (2 entrées d'accélération vers les états ax, ay)
        self.B = np.zeros((8, 2))
        self.B[4, 0] = 1.0
        self.B[5, 1] = 1.0
        
        # Matrice de transition cinématique de base (sera mise à jour avec dt)
        self.F = np.eye(8)
        self.F[6, 6] = 1.0 # Persistance biais x
        self.F[7, 7] = 1.0 # Persistance biais y

    def predict(self, u_local, dt):
        """ Étape de propagation locale (Inertie pure) """
        # Mise à jour de la matrice F avec le pas de temps actuel
        self.F[0, 2] = dt
        self.F[0, 4] = 0.5 * dt**2
        self.F[1, 3] = dt
        self.F[1, 5] = 0.5 * dt**2
        self.F[2, 4] = dt
        self.F[3, 5] = dt
        
        # Propagation de l'état et de la covariance
        self.x_est = self.F @ self.x_est + self.B @ u_local
        self.P_est = self.F @ self.P_est @ self.F.T + self.Q

    def update_local_sensor(self, z_local, sensor_type, R_local):
        """ Étape de mise à jour classique pour les capteurs embarqués directs """
        I = np.eye(8)
        
        if sensor_type == 'GPS':
            # Le GPS n'observe que x et y (Drone 1)
            H = np.zeros((2, 8))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            innov = z_local - H @ self.x_est
            
        elif sensor_type == 'IMU':
            # L'accéléromètre observe ax + bx et ay + by (Drone 2)
            H = np.zeros((2, 8))
            H[0, 4] = 1.0; H[0, 6] = 1.0
            H[1, 5] = 1.0; H[1, 7] = 1.0
            z_pred = np.array([self.x_est[4] + self.x_est[6], 
                               self.x_est[5] + self.x_est[7]])
            innov = z_local - z_pred
        else:
            return

        S = H @ self.P_est @ H.T + R_local
        K = self.P_est @ H.T @ np.linalg.inv(S)
        
        self.x_est = self.x_est + K @ innov
        self.P_est = (I - K @ H) @ self.P_est

    def update_inter_drone_distance(self, d_measured, neighbor_x, neighbor_P, R_dist):
        """ 
        Mise à jour distribuée par INTERSECTION DE COVARIANCE (CI)
        Reçoit l'état transmis par le voisin via le réseau de communication.
        """
        # Extraction des positions estimées locales et distantes pour linéarisation
        xi, yi = self.x_est[0], self.x_est[1]
        xj, yj = neighbor_x[0], neighbor_x[1]
        
        d_pred = np.sqrt((xi - xj)**2 + (yi - yj)**2)
        if d_pred < 1e-4: d_pred = 1e-4  # Sécurité division par zéro
        
        # Jacobien H_i par rapport à l'état du drone local (i)
        H_i = np.zeros((1, 8))
        H_i[0, 0] = (xi - xj) / d_pred
        H_i[0, 1] = (yi - yj) / d_pred
        
        # Jacobien H_j par rapport à l'état du drone voisin (j)
        H_j = np.zeros((1, 8))
        H_j[0, 0] = -(xi - xj) / d_pred
        H_j[0, 1] = -(yi - yj) / d_pred
        
        innov = d_measured - d_pred

        # Définition de la fonction de coût pour trouver le omega optimal (minimisation de la trace)
        def covariance_intersection_cost(omega):
            P_scaled_i = self.P_est / omega
            P_scaled_j = neighbor_P / (1.0 - omega)
            S = H_i @ P_scaled_i @ H_i.T + H_j @ P_scaled_j @ H_j.T + R_dist
            K = P_scaled_i @ H_i.T / S[0, 0]
            P_up = P_scaled_i - K @ H_i @ P_scaled_i
            return np.trace(P_up)

        # Optimisation de omega bornée strictement entre 0 et 1
        sol = minimize_scalar(covariance_intersection_cost, bounds=(0.01, 0.99), method='bounded')
        omega_opt = sol.x

        # Application des équations finales du EKF-CI avec le omega optimal
        P_scaled_i = self.P_est / omega_opt
        P_scaled_j = neighbor_P / (1.0 - omega_opt)
        
        S_opt = H_i @ P_scaled_i @ H_i.T + H_j @ P_scaled_j @ H_j.T + R_dist
        K_opt = P_scaled_i @ H_i.T / S_opt[0, 0]
        
        # Mise à jour de l'état local et de sa covariance interne
        self.x_est = self.x_est + (K_opt * innov).flatten()
        self.P_est = P_scaled_i - K_opt @ H_i @ P_scaled_i


# =====================================================================
# EXEMPLE DE SIMULATION DU RÉSEAU COMMUNICATION / BOUCLE TEMPORELLE
# =====================================================================
if __name__ == "__main__":
    dt = 0.1
    
    # Configuration des matrices initiales (dim 8x8)
    P_init = np.eye(8) * 1.0
    Q_local = np.eye(8) * 0.01
    Q_local[0:4, 0:4] = 1e-4  # Triche EKF sur la cinématique
    
    # Instanciation de notre essaim distribué
    drone1 = DistributedDrone(drone_id=1, x_init=np.array([0, 10, 1, 0, 0, 0, 0, 0]), P_init=P_init, Q_local=Q_local)
    drone2 = DistributedDrone(drone_id=2, x_init=np.array([10, 0, 1, 0, 0, 0, 0, 0]), P_init=P_init, Q_local=Q_local)
    drone3 = DistributedDrone(drone_id=3, x_init=np.array([0, -10, 1, 0, 0, 0, 0, 0]), P_init=P_init, Q_local=Q_local)
    
    essaim = [drone1, drone2, drone3]
    
    # --- SIMULATION D'UN PAS DE TEMPS ---
    # 1. Étape de Prédiction Locale (Chaque drone travaille de son côté)
    u_fake = np.array([0.0, 2.0])
    for drone in essaim:
        drone.predict(u_local=u_fake, dt=dt)
        
    # 2. Étape des capteurs locaux directs
    # Le Drone 1 met à jour son GPS
    R_gps = np.eye(2) * 0.01
    z_gps_drone1 = np.array([0.05, 10.02])
    drone1.update_local_sensor(z_local=z_gps_drone1, sensor_type='GPS', R_local=R_gps)
    
    # Le Drone 2 met à jour son Accéléromètre bruité
    R_imu = np.eye(2) * 0.04
    z_imu_drone2 = np.array([0.48, 1.95]) # Contient acc + biais
    drone2.update_local_sensor(z_local=z_imu_drone2, sensor_type='IMU', R_local=R_imu)

    # 3. Étape de Télémétrie Collaborative (Échange réseau simulé)
    # Le Drone 2 mesure une distance avec le Drone 1.
    # Dans un vrai système, le Drone 1 envoie sa "Payload" par radio : (x_est, P_est)
    d_mesure_12 = 14.14 # Exemple de distance vraie bruitée
    R_dist = np.array([[0.01]])
    
    drone2.update_inter_drone_distance(
        d_measured=d_mesure_12, 
        neighbor_x=drone1.x_est, 
        neighbor_P=drone1.P_est, 
        R_dist=R_dist
    )
    
    print("Mise à jour distribuée réussie !")
    print(f"Position estimée distribuée du Drone 2 : X={drone2.x_est[0]:.2f}, Y={drone2.x_est[1]:.2f}")
