import torch 
from torch.utils.data import Dataset
import numpy as np
from .config import Config as config 

def get_rotation_matrix(phi, theta, psi):

    R_x = np.array(
    [[1, 0, 0],
    [0, np.cos(phi), -np.sin(phi)],
    [0, np.sin(phi), np.cos(phi)]])

    R_y = np.array(
    [[np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]])

    R_z = np.array(
    [[np.cos(psi), -np.sin(psi), 0],
    [np.sin(psi), np.cos(psi), 0],
    [0, 0, 1]])

    return R_z @ R_y @ R_x

class TrajectoryDataset(Dataset):

    def __init__(self, num_trajectories, type_data, n_drones, config):
        super().__init__()
        self.num_trajectories = num_trajectories
        self.seq_len = config.seq_len
        self.dt = config.dt
        self.config = config
        self.type_data = type_data
        self.dataset_folder = config.dataset_folder
        self.n_drones = n_drones
        self.g = 9.81

        # Adressage des colonnes : 
        # 0, 1, 2 : Vérité - Position (x, y, z)
        # 3, 4, 5 : Vérité - Vitesses (vx, vy, vz)
        # 6, 7, 8 : Vérité - Angles d'Euler (phi, theta, psi)
        # 9, 10, 11 : Capteur - IMU Accéléromètre (ax, ay, az en repère drone)
        # 12, 13, 14 : Capteur - IMU Gyroscope (p, q, r en repère drone)
        # 15, 16, 17 : Capteur - GPS
        # 18 : Temps
        # 19+ : Distances vers les autres drones
        
        self.base_features = 19
        self.num_features = self.base_features + (self.n_drones - 1)
        self.data = torch.zeros((int(num_trajectories), int(self.seq_len), self.n_drones, self.num_features)).to(dtype=torch.float32) 

        print(f"Génération d'un essaim à {self.n_drones} drones")
        
        # Corrigé : Initialisation à None pour que la sauvegarde ne plante pas
        self.init_states = None 

        self._generate_data(config)

        if config.IS_SAVE_DATASET:
            print("Sauvegarde des données en cours...")
            self.save_dataset(config.dataset_filename)

        if config.IS_NORMALISATION:
            print("   Normalisation des capteurs en cours...")
            self.__normalize(config)
        

    def _generate_data(self, config):

        for i in range(int(self.num_trajectories)):
            # Positions étendues pour que le drone parcourt de vraies distances
            pos = np.random.uniform(low=[0, 0, 10], high=[100, 100, 50], size=(self.n_drones, 3))

            vel = np.random.uniform(-2, 2, size=(self.n_drones, 3))

            angles = np.random.uniform(-0.1, 0.1, size=(self.n_drones, 3))
            angles[:, 2] = np.random.uniform(-np.pi, np.pi, size=self.n_drones)

            angle_vel = np.random.uniform(-0.05, 0.05, size=(self.n_drones, 3))

            biais_acc = np.random.normal(0, 0.02, size=(self.n_drones, 3))

            for t in range(self.seq_len):
                vel_prev = vel.copy()

                accel_cmd = np.random.normal(0, 0.5, size=(self.n_drones, 3))
                ang_accel_cmd = np.random.normal(0, 0.1, size=(self.n_drones, 3))

                # Physique de base (intégration d'Euler)
                vel += accel_cmd * self.dt
                pos += vel * self.dt
                angle_vel += ang_accel_cmd * self.dt
                angles += angle_vel * self.dt

                # Accélération terrestre réelle
                a_earth = (vel - vel_prev) / self.dt

                for d in range(self.n_drones):
                    self.data[i, t, d, 0:3] = torch.tensor(pos[d])
                    self.data[i, t, d, 3:6] = torch.tensor(vel[d])
                    self.data[i, t, d, 6:9] = torch.tensor(angles[d])

                    R = get_rotation_matrix(angles[d, 0], angles[d, 1], angles[d, 2])
                    R_inv = R.T

                    grav_vec = np.array([0, 0, self.g])
                    a_body_pure = R_inv @ (a_earth[d] + grav_vec)
                    g_body_bruit = a_body_pure + biais_acc[d] + np.random.normal(0, 0.1, 3)
                    self.data[i, t, d, 9:12] = torch.tensor(g_body_bruit)
                    
                    # CORRECTION : Ajout du gyroscope (indices 12 à 14)
                    gyro_bruit = angle_vel[d] + np.random.normal(0, 0.05, 3)
                    self.data[i, t, d, 12:15] = torch.tensor(gyro_bruit)

                    # CORRECTION : L'indentation était mauvaise, ceci doit être dans la boucle 'for d'
                    gps_bruit = pos[d] + np.random.normal(0, 2.0, 3)
                    self.data[i, t, d, 15:18] = torch.tensor(gps_bruit)
                    self.data[i, t, d, 18] += self.dt 
                    
                    dist_idx = self.base_features
                    for autre_d in range(self.n_drones):
                        # CORRECTION : L'opérateur était := au lieu de !=
                        if d != autre_d: 
                            vraie_dist = np.linalg.norm(pos[d] - pos[autre_d])
                            mesure_dist = vraie_dist + np.random.normal(0, 0.5)
                            self.data[i, t, d, dist_idx] = mesure_dist
                            dist_idx += 1

        print(f"   Nombre de trajectoires générées : {self.num_trajectories} ")
  

    def __normalize(self, config):
        s_pos = config.scale_pos
        s_acc = config.scale_acc
        s_gyro = config.scale_gyro
        s_dist = config.scale_dist

        # CORRECTION 1 : On ne touche PAS à 0:9 (la vérité terrain doit rester en vraies unités)
        # CORRECTION 2 : Ajout de la dimension des drones (:, :, :, idx)

        # On normalise uniquement les capteurs
        self.data[:, :, :, 9:12] /= s_acc          # Accéléromètre
        self.data[:, :, :, 12:15] /= s_gyro        # Gyroscope
        self.data[:, :, :, 15:18] /= s_pos         # GPS

        if self.n_drones > 1:
            self.data[:, :, :, self.base_features:] /= s_dist # Distances

        # On "clamp" (bride) uniquement les valeurs des capteurs, pas la vérité
        self.data[:, :, :, 9:] = torch.clamp(self.data[:, :, :, 9:], min=-5.0, max=5.0)

    def __len__(self):
        return int(self.num_trajectories)

    def __getitem__(self, idx):
        trajectoire = self.data[idx]
        etat_initial = trajectoire[0, :, 0:9].clone()
        return etat_initial, trajectoire

    def save_dataset(self, filename):
        # Sauvegarde sécurisée
        data_to_save = {'data': self.data, 'config': self.config}
        if self.init_states is not None:
            data_to_save['init_states'] = self.init_states
            
        filename = self.dataset_folder + "/" + self.type_data + "_" + filename
        torch.save(data_to_save, filename)
        print(f"Le Dataset a bien été enregistré : '{filename}'.")


import torch 
from torch.utils.data import Dataset
import numpy as np
from .config import Config as config 

def get_rotation_matrix(phi, theta, psi):

    R_x = np.array(
    [[1, 0, 0],
    [0, np.cos(phi), -np.sin(phi)],
    [0, np.sin(phi), np.cos(phi)]])

    R_y = np.array(
    [[np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]])

    R_z = np.array(
    [[np.cos(psi), -np.sin(psi), 0],
    [np.sin(psi), np.cos(psi), 0],
    [0, 0, 1]])

    return R_z @ R_y @ R_x

class TrajectoryDataset(Dataset):

    def __init__(self, num_trajectories, type_data, n_drones, config):
        super().__init__()
        self.num_trajectories = num_trajectories
        self.seq_len = config.seq_len
        self.dt = config.dt
        self.config = config
        self.type_data = type_data
        self.dataset_folder = config.dataset_folder
        self.n_drones = n_drones
        self.g = 9.81


        #Adressage des colonnes : 
        #0, 1, 2 : vérité - Position (x, y, z)
        #3, 4, 5 : vérité - Vitesses (vx, vy, vz)
        #6, 7, 8 : verité - Angles d'Euler (phi, theta, psi)
        #9, 1O, 11 :  Capteur - IMU Accéléromètre (ax, ay, az en repere drone)
        #12, 13, 14 : Capteur - IMU Gyroscope (p, q, r en repère drone)
        #15, 16, 17 : Capteur - Distances vers les (N-1) autres drones
        self.base_features = 19
        self.num_features = self.base_features + (self.n_drones - 1)
        self.data = torch.zeros((int(num_trajectories), int(self.seq_len), self.n_drones, self.num_features)).to(dtype=torch.float32) # Pourquoi 13 est codé en dur

        print(f"Génération d'un essaim à {self.n_drones} drones")
        #self.init_states = torch.zeros((int(num_trajectories), 5)).to(dtype=torch.float32) #Pourquoi 5 est codé en dur ?

        self._generate_data(config)

        if config.IS_SAVE_DATASET:
            print("Sauvegarde des données en cours...")
            self.save_dataset(config.dataset_filename)

        if config.IS_NORMALISATION :
            print("   Normalisation en cours...")
            self.__normalize(config)
        


    def _generate_data(self, config):

        for i in range(int(self.num_trajectories)):
            pos = np.random.uniform(low=[0, 0, 10], high=[100, 100, 50], size=(self.n_drones, 3))

            vel = np.random.uniform(-2, 2, size=(self.n_drones, 3))

            angles = np.random.uniform(-0.1, 0.1, size=(self.n_drones, 3))
            angles[:,2] = np.random.uniform(-np.pi, np.pi, size=self.n_drones)

            angle_vel = np.random.uniform(-0.05, 0.05, size=(self.n_drones, 3))

            biais_acc = np.random.normal(0, 0.02, size=(self.n_drones, 3))

            for t in range(self.seq_len):
                vel_prev = vel.copy()

                accel_cmd = np.random.normal(0, 0.5, size=(self.n_drones, 3))
                ang_accel_cmd = np.random.normal(0, 0.1, size=(self.n_drones, 3))

                #Physique de base (intégration d'euler)
                vel += accel_cmd * self.dt
                pos += vel * self.dt
                angle_vel += ang_accel_cmd * self.dt
                angles += angle_vel * self.dt

                #Acceleration terreste réelle
                a_earth = (vel -vel_prev) / self.dt

                for d in range(self.n_drones):
                    self.data[i, t, d, 0:3] = torch.tensor(pos[d])
                    self.data[i, t, d, 3:6] = torch.tensor(vel[d])
                    self.data[i, t, d, 6:9] = torch.tensor(angles[d])

                    R = get_rotation_matrix(angles[d, 0], angles[d, 1], angles[d,2])

                    R_inv = R.T

                    grav_vec = np.array([0, 0, self.g])
                    a_body_pure = R_inv @ (a_earth[d] + grav_vec)
                    g_body_bruit = a_body_pure + biais_acc[d] + np.random.normal(0, 0.1, 3)
                    self.data[i, t, d, 9:12] = torch.tensor(g_body_bruit)

                    gps_bruit = pos[d] + np.random.normal(0, 2.0, 3)

                self.data[i, t, d, 15:18] = torch.tensor(gps_bruit)

                self.data[i, t, d, 18] += self.dt 
                dist_idx = self.base_features
                for autre_d in range(self.n_drones):
                    if d := autre_d:
                        vraie_dist = np.linalg.norm(pos[d] - pos[autre_d])
                        mesure_dist = vraie_dist + np.random.normal(0, 0.5)
                        self.data[i, t, d, dist_idx] = mesure_dist
                        dist_idx +=1

        print(f"   Nombre de trajectoires générées : {self.num_trajectories} ")
  

    def __normalize(self, config):
        s_pos = config.scale_pos
        s_vel = config.scale_vel
        s_acc = config.scale_acc
        s_ang = config.scale_angle
        s_gyro = config.scale_gyro
        s_dist = config.scale_dist

        s_acc_ang = config.scale_acc_angle

        self.data[:, :, 0:3] /=s_pos
        self.data[:, :, 15:18] /=s_pos

        self.data[:, :, 3:6] /= s_vel

        self.data[:, :, 6:9] /= s_ang

        self.data[:, :, 9:12] /= s_acc

        self.data[:, :, 12:15] /= s_gyro

        if self.n_drones > 1 :
            self.data[:, :, :, self.base_features:] /= s_dist

        self.data = torch.clamp(self.data, min=-5.0, max=5.0)

    def __len__(self):
        return int(self.num_trajectories)

    def __getitem__(self, idx):
        trajectoire = self.data[idx]
        etat_initial = trajectoire[0, :, 0:9].clone()
        return etat_initial, trajectoire

    def save_dataset(self, filename):
        data_to_save = {'data' : self.data, 'init_states' : self.init_states, 'config' : self.config}
        filename = self.dataset_folder + "/" + self.type_data + "_" + filename
        torch.save(data_to_save, filename)
        print(f"Le Dataset a bien été enregistré : '{filename}'.")
