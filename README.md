import matplotlib.pyplot as plt 
import numpy as np
from src.config import Config as config

def plot_trajectory(vrai_etat, pred_etat, mesures_gps, trajectory_id=1):
    """
    Trace l'évolution des 9 variables d'état dans le temps sur une seule image.
    Attend des tableaux (numpy) de forme (seq_len, 9) pour les états et (seq_len, 3) pour le GPS.
    """
    # Création d'une grille de 3 lignes x 3 colonnes
    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    fig.suptitle(f"Évaluation des Variables d'État - Trajectoire n°{trajectory_id}", fontsize=16, fontweight='bold')

    noms_variables = [
        "Position X (m)", "Position Y (m)", "Position Z (m)",
        "Vitesse X (m/s)", "Vitesse Y (m/s)", "Vitesse Z (m/s)",
        "Angle Phi - Roulis (rad)", "Angle Theta - Tangage (rad)", "Angle Psi - Lacet (rad)"
    ]

    # Création d'un axe temporel (de 0 à seq_len)
    t = range(len(vrai_etat))

    for i in range(9):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        # 1. Trace la vérité terrain (Vert)
        ax.plot(t, vrai_etat[:, i], 'g-', label='Vérité terrain', linewidth=2)

        # 2. Trace les prédictions KalmanNet (Bleu pointillé)
        ax.plot(t, pred_etat[:, i], 'b--', label='KalmanNet', linewidth=1.5)

        # 3. Trace les mesures GPS (Rouge) SEULEMENT pour les positions (indices 0, 1, 2)
        if mesures_gps is not None and i < 3:
            ax.scatter(t, mesures_gps[:, i], c='red', s=15, alpha=0.4, label='Mesures GPS')

        # Mise en forme du sous-graphique
        ax.set_title(noms_variables[i])
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Pas de temps")
        
        # On met la légende uniquement sur le premier graphique pour ne pas surcharger l'image
        if i == 0:
            ax.legend()

    # Ajuste l'espacement pour que les titres ne se chevauchent pas
    plt.tight_layout()
    plt.subplots_adjust(top=0.9) 
    plt.show()

def plot_training_loss(epoch, train_losses, val_losses):
    """
    Trace l'évolution de la loss pendant l'entraînement
    """
    plt.figure(figsize=(10,5))
    plt.plot(epoch, train_losses, label="Train Loss")
    if val_losses:
        plt.plot(epoch, val_losses, label="Val Loss", color='orange')

    plt.title("Courbe d'apprentissage du KalmanNet")
    plt.xlabel("Epochs")
    plt.ylabel(f"Erreur Loss : {config.name_loss}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
