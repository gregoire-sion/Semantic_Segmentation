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
            if i>=3: break

            batch_init = batch_init.to(config.device)
            batch_data = batch_data.to(config.device)

            predictions = model(batch_init, batch_data)

            vrai_x = batch_data[0, :,0].cpu().numpy()
            vrai_y = batch_data[0, :,2].cpu().numpy()
            gps_x = batch_data[0, :,11].cpu().numpy()
            gps_y = batch_data[0, :,12].cpu().numpy()
            pred_x = predictions[0, :,0].cpu().numpy()
            pred_y = predictions[0, :,2].cpu().numpy()

            plot_trajectory(vrai_x, vrai_y, gps_x, gps_y, pred_x, pred_y, trajectory_id=i+1)
    
    print("Evaluation terminée")
