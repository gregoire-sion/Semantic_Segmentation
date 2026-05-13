<img width="1878" height="961" alt="image" src="https://github.com/user-attachments/assets/f00a1c25-2340-4457-9399-33b772d5f6d7" />

import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_swarm_3d(verite_terrain, kalman_pred, titre="Trajectoire 3D de l'Essaim"):
    """
    Affiche la trajectoire 3D de tous les drones.
    
    Args:
        verite_terrain: Array ou Tenseur de shape (seq_len, n_drones, features)
        kalman_pred: Array ou Tenseur de shape (seq_len, n_drones, features)
    """
    # 1. Sécurité : Conversion en numpy si ce sont des tenseurs PyTorch
    if torch.is_tensor(verite_terrain):
        verite_terrain = verite_terrain.detach().cpu().numpy()
    if torch.is_tensor(kalman_pred):
        kalman_pred = kalman_pred.detach().cpu().numpy()
        
    seq_len, n_drones, _ = verite_terrain.shape
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Génération d'une palette de couleurs distinctes pour chaque drone
    colors = plt.cm.tab10(np.linspace(0, 1, n_drones))
    
    for d in range(n_drones):
        # L'index 0, 1 et 2 correspondent aux positions X, Y, Z
        gt_x = verite_terrain[:, d, 0]
        gt_y = verite_terrain[:, d, 1]
        gt_z = verite_terrain[:, d, 2]
        
        pred_x = kalman_pred[:, d, 0]
        pred_y = kalman_pred[:, d, 1]
        pred_z = kalman_pred[:, d, 2]
        
        # Tracé de la Vérité Terrain (Ligne continue)
        ax.plot(gt_x, gt_y, gt_z, color=colors[d], linewidth=2.5, 
                label=f'Vérité Terrain - Drone {d+1}')
        
        # Tracé des Prédictions Kalman (Ligne pointillée, avec des repères)
        ax.plot(pred_x, pred_y, pred_z, color=colors[d], linestyle='--', 
                linewidth=1.5, marker='o', markersize=3, markevery=10, alpha=0.8,
                label=f'KalmanNet - Drone {d+1}')
        
        # Marqueurs de Départ (Croix) et d'Arrivée (Étoile)
        ax.scatter(gt_x[0], gt_y[0], gt_z[0], color=colors[d], marker='X', s=100)
        ax.scatter(gt_x[-1], gt_y[-1], gt_z[-1], color=colors[d], marker='*', s=150)

    # Cosmétique et labels
    ax.set_xlabel('Position X (m)', fontsize=11)
    ax.set_ylabel('Position Y (m)', fontsize=11)
    ax.set_zlabel('Position Z (m)', fontsize=11)
    ax.set_title(titre, fontsize=16, fontweight='bold')
    
    # 2. Conservation des proportions 1:1:1 pour la physique
    # Sans ça, Matplotlib écrase les axes et fausse la perception des distances
    all_data = np.concatenate([verite_terrain[..., 0:3], kalman_pred[..., 0:3]], axis=0)
    max_range = np.array([all_data[..., 0].max() - all_data[..., 0].min(), 
                          all_data[..., 1].max() - all_data[..., 1].min(), 
                          all_data[..., 2].max() - all_data[..., 2].min()]).max() / 2.0

    mid_x = (all_data[..., 0].max() + all_data[..., 0].min()) * 0.5
    mid_y = (all_data[..., 1].max() + all_data[..., 1].min()) * 0.5
    mid_z = (all_data[..., 2].max() + all_data[..., 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Légende placée à l'extérieur pour ne pas cacher la trajectoire
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.show()

