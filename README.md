import torch
from torch.utils.data import DataLoader
from src.dataset import TrajectoryDataset
from src.models.kalmannet import KalmanNet
from plots import plot_trajectory

def run_testing(config):

    test_set = TrajectoryDataset(config.n_test, "test", config=config)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    model = KalmanNet(config).to(config.device)
    model.load_state_dict(torch.load(config.model_path, map_location=config.device))
    model.eval()

    with torch.no_grad():
        for i, (batch_init, batch_data) in enumerate(test_loader) : 
            if i >= 3: break

            batch_init = batch_init.to(config.device)
            batch_data = batch_data.to(config.device)

            predictions = model(batch_init, batch_data)

            # --- NOUVELLE EXTRACTION ---
            # On prend : Batch 0, Tout le temps (:), Drone 0, et les colonnes correspondantes
            
            # 1. Vérité terrain : Les 9 premières colonnes (x, y, z, vx, vy, vz, phi, theta, psi)
            vrai_etat = batch_data[0, :, 0, 0:9].cpu().numpy()
            
            # 2. Prédictions : Les 9 variables de sortie de ton modèle
            pred_etat = predictions[0, :, 0, 0:9].cpu().numpy()
            
            # 3. GPS : Les 3 colonnes du GPS (Assumé 15, 16, 17 selon notre config précédente)
            # Si dans ton dataset actuel le GPS commence à l'index 11, remplace par [0, :, 0, 11:14]
            mesures_gps = batch_data[0, :, 0, 15:18].cpu().numpy()

            # Appel de la nouvelle fonction
            plot_trajectory(vrai_etat, pred_etat, mesures_gps, trajectory_id=i+1)
    
    print("Evaluation terminée")
