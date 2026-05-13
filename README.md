import matplotlib.pyplot as plt 
from src.config import Config as config

def plot_trajectory(vrai_x, vrai_y, gps_x, gps_y, pred_x, pred_y, pred_x_ekf, pred_y_ekf, trajectory_id=1):
    """
    Trace la trajectoire réelle, les mesures GPS et les prédictions du filtre.
    """
    plt.figure(figsize=(10,6))

    plt.plot(vrai_x, vrai_y, 'g-', label='Verité terrain', linewidth=2)

    plt.scatter(gps_x, gps_y, c='red', s=10, alpha=0.3, label='Mesures GPS (bruitées)')

    plt.plot(pred_x, pred_y, 'b--', label='Estimation KalmanNet', linewidth=1.5)

    plt.title(f"Evaluation de la Trajectoire n°{trajectory_id}")
    plt.xlabel("Position X(m)")
    plt.ylabel("Position Y(m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.show()

def plot_training_loss(epoch, train_losses, val_losses):
    """
    Trace l'evolution de la loss pendant l'entrainement
    """
    plt.figure(figsize=(10,5))
    plt.plot(epoch, train_losses, label="Train Loss")
    if val_losses:
        plt.plot(epoch, val_losses, label="Val Loss", color='orange')

    plt.title("Courbe d'apprenstisage du KalmanNet")
    plt.xlabel("Epochs")
    plt.ylabel(f"Erreur Loss : {config.name_loss}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
