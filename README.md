import matplotlib.pyplot as plt
import numpy as np

# --- (Ton extraction initiale des variables reste identique) ---

# Fonction utilitaire pour générer la figure d'un drone spécifique
def tracer_figure_drone(id_drone, indices_ekf, etats_vrais, mesures_xy=None):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Analyse EKF - Drone {id_drone}", fontsize=14, fontweight='bold')
    axs = axs.flatten()
    
    # Les 4 sous-états pour un drone
    labels_locaux = ["Position X", "Position Y", "Vitesse Vx", "Vitesse Vy"]
    
    for idx, (idx_ekf, etat_vrai, label) in enumerate(zip(indices_ekf, etats_vrais, labels_locaux)):
        ax = axs[idx]
        
        # Récupération EKF et Covariance
        etat_estime = traj_kalman[:, 0, idx_ekf]
        sigma = np.sqrt(P_hist_np[:, idx_ekf, idx_ekf])
        
        # Tracés EKF et Couloir 3-sigma
        ax.plot(Temps_np, etat_estime, color='blue', label='Estimation EKF')
        ax.fill_between(Temps_np, etat_estime - 3*sigma, etat_estime + 3*sigma, 
                        color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
        
        # Tracé de la vérité terrain
        ax.plot(Temps_np, etat_vrai, color='black', linestyle='--', label='Vérité')
        
        # Ajout des mesures capteurs (uniquement pour X et Y, soit idx < 2)
        if mesures_xy is not None and idx < 2:
            ax.scatter(temps_capteur_np, mesures_xy[idx], color="red", marker="x", linewidths=0.5, label='Mesure')
            
        # Esthétique
        ax.set_title(f"{label} ({noms_etats[idx_ekf]})", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)
        if idx >= 2: # Ligne du bas
            ax.set_xlabel("Temps (s)")
            
    # Légende unique pour la figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.95))
    plt.tight_layout()
    plt.subplots_adjust(top=0.85) # Laisse de la place au titre et à la légende

# ---------------------------------------------------------
# 1. TRACÉ DRONE 1
# Indices : X=0, Y=3, Vx=6, Vy=9
# ---------------------------------------------------------
tracer_figure_drone(1, [0, 3, 6, 9], 
                    [x_vrai_1, y_vrai_1, vx_vrai_1, vy_vrai_1], 
                    mesures_xy=[x1_mesure, y1_mesure])

# ---------------------------------------------------------
# 2. TRACÉ DRONE 2
# Indices : X=1, Y=4, Vx=7, Vy=10
# ---------------------------------------------------------
tracer_figure_drone(2, [1, 4, 7, 10], 
                    [x_vrai_2, y_vrai_2, vx_vrai_2, vy_vrai_2], 
                    mesures_xy=[x2_mesure, y2_mesure])

# ---------------------------------------------------------
# 3. TRACÉ DRONE 3
# Indices : X=2, Y=5, Vx=8, Vy=11
# ---------------------------------------------------------
tracer_figure_drone(3, [2, 5, 8, 11], 
                    [x_vrai_3, y_vrai_3, vx_vrai_3, vy_vrai_3], 
                    mesures_xy=None) # Pas de mesures extraites pour le drone 3

# ---------------------------------------------------------
# 4. TRACÉ DES BIAIS (États 12 à 17)
# ---------------------------------------------------------
fig_biais, axs_biais = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
fig_biais.suptitle("Analyse EKF - États globaux (Biais)", fontsize=14, fontweight='bold')
axs_biais = axs_biais.flatten()

for idx, idx_ekf in enumerate(range(12, 18)):
    ax = axs_biais[idx]
    etat_estime = traj_kalman[:, 0, idx_ekf]
    sigma = np.sqrt(P_hist_np[:, idx_ekf, idx_ekf])
    
    ax.plot(Temps_np, etat_estime, color='blue', label='Estimation EKF')
    ax.fill_between(Temps_np, etat_estime - 3*sigma, etat_estime + 3*sigma, 
                    color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    
    ax.set_title(noms_etats[idx_ekf], fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.7)
    if idx >= 4:
        ax.set_xlabel("Temps (s)")

handles_b, labels_b = axs_biais[0].get_legend_handles_labels()
fig_biais.legend(handles_b, labels_b, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.95))
plt.tight_layout()
plt.subplots_adjust(top=0.88)

plt.show()




temps_np = np.array(temps)[:traj.shape[0]]
temps_capteur_np = np.array(temps_capteur)[:traj.shape[0]]
P_hist_np = np.array(P_historique)[:traj.shape[0]]

traj = traj[:-1, :, :]

traj_kalman = traj_kalman[:-1, :, :]

x_vrai_1 = traj[:, 0, 0]
y_vrai_1 = traj[:, 0, 1]
vx_vrai_1 = traj[:, 0, 2]
vy_vrai_1 = traj[:, 0, 3]

x_vrai_2 = traj[:, 1, 0]
y_vrai_2 = traj[:, 1, 1]
vx_vrai_2 = traj[:, 1, 2]
vy_vrai_2 = traj[:, 1, 3]

x_vrai_3 = traj[:, 2, 0]
y_vrai_3 = traj[:, 2, 1]
vx_vrai_3 = traj[:, 2, 2]
vy_vrai_3 = traj[:, 2, 3]

x_capteur_2 = traj[:, 3, 0]
y_capteur_2 = traj[:, 3, 1]

x_corrige_2 = traj[:, 4, 0]
y_corrige_2 = traj[:, 4, 1]

x_kalman_1 = traj_kalman[:, 0, 0]
y_kalman_1 = traj_kalman[:, 0, 1]

x_kalman_2 = traj_kalman[:, 0, 4]
print(f"Version kalman : {x_kalman_2}")
y_kalman_2 = traj_kalman[:, 0, 5]

x_kalman_3 = traj_kalman[:, 0, 8]
y_kalman_3 = traj_kalman[:, 0, 9]

bx_kalman = traj_kalman[:, 0, 12]
by_kalman = traj_kalman[:, 0, 13]

derive_theorique_x = np.ones(int(t_max/dt-1)) * derive_x
derive_theorique_y = np.ones(int(t_max/dt-1)) * derive_y

x1_mesure = mesures_capteur[:, 0, 0]
y1_mesure = mesures_capteur[:, 0, 1]

x2_mesure = mesures_capteur[:, 0, 2]
y2_mesure = mesures_capteur[:, 0, 3]

noms_etats = ["X Drone 1", "X Drone 2", "X Drone 3", "Y Drone 1", "Y Drone 2", "Y Drone 3","Vx Drone 1", "Vx Drone 2", "Vx Drone 3","Vy Drone 1", "Vy Drone 2", "Vy Drone 3", "Biais x", "Biais y", "Biais Vx", "Biais Vy", "Biais ax", "Biais ay"]

fig, axs = plt.subplots(7,3, figsize=(18,20), sharex=True)
# axs = axs.flatten()

#Drone1
j=0
for i in [0, 4, 8]:
    etat_estime = traj_kalman[:, 0, i] 
    variance = P_hist_np[:, i, i]
    sigma = np.sqrt(variance)
    borne_haute = etat_estime + 3 * sigma
    borne_basse = etat_estime - 3 * sigma
    
    axs[j,0].plot(temps_np, etat_estime, color='blue', label='Estimation EKF')
    
    axs[j,0].fill_between(temps_np, borne_basse, borne_haute, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    
    # Esthétique du graphique
    axs[j,0].set_title(noms_etats[i], fontsize=10, fontweight='bold')
    axs[j,0].grid(True, linestyle=':', alpha=0.7)
    j+=1

k=0
#Drone2
for i in [1, 5, 9]:
    etat_estime = traj_kalman[:, 0, i] 

    variance = P_hist_np[:, i, i]
    sigma = np.sqrt(variance)
    borne_haute = etat_estime + 3 * sigma
    borne_basse = etat_estime - 3 * sigma
    
    axs[k,1].plot(temps_np, etat_estime, color='blue', label='Estimation EKF')
    
    axs[k,1].fill_between(temps_np, borne_basse, borne_haute, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    
    # Esthétique du graphique
    axs[k,1].set_title(noms_etats[i], fontsize=10, fontweight='bold')
    axs[k,1].grid(True, linestyle=':', alpha=0.7)
    k+=1

l=0
#Drone3
for i in [2, 6, 10]:
    etat_estime = traj_kalman[:, 0, i] 

    variance = P_hist_np[:, i, i]
    sigma = np.sqrt(variance)
    borne_haute = etat_estime + 3 * sigma
    borne_basse = etat_estime - 3 * sigma
    
    axs[l,2].plot(temps_np, etat_estime, color='blue', label='Estimation EKF')

    axs[l,2].fill_between(temps_np, borne_basse, borne_haute, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    
    # Esthétique du graphique
    axs[l,2].set_title(noms_etats[i], fontsize=10, fontweight='bold')
    axs[l,2].grid(True, linestyle=':', alpha=0.7)
    l+=1

m=0
n=4
for i in range(11 ,18):
    if m > 2:
        m=0
        n+=1

    etat_estime = traj_kalman[:, 0, i] 

    variance = P_hist_np[:, i, i]
    sigma = np.sqrt(variance)
    borne_haute = etat_estime + 3 * sigma
    borne_basse = etat_estime - 3 * sigma

    print(f"Dimensions de axs: {axs.shape}")
    print(f"Indices n et m: {n}, {m}")
    
    axs[n,m].plot(temps_np, etat_estime, color='blue', label='Estimation EKF')
    
    axs[n,m].fill_between(temps_np, borne_basse, borne_haute, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    
    # Esthétique du graphique
    axs[n,m].set_title(noms_etats[i], fontsize=10, fontweight='bold')
    axs[n,m].grid(True, linestyle=':', alpha=0.7)
    m+=1

axs[0,0].plot(temps_np, x_vrai_1, color='black', linestyle='--', label='Vérité')
axs[1,0].plot(temps_np, y_vrai_1, color='black', linestyle='--', label='Vérité')
axs[2,0].plot(temps_np, vx_vrai_1, color='black', linestyle='--', label='Vérité')
axs[3,0].plot(temps_np, vy_vrai_1, color='black', linestyle='--', label='Vérité')

axs[0,1].plot(temps_np, x_vrai_2, color='black', linestyle='--', label='Vérité')
axs[1,1].plot(temps_np, y_vrai_2, color='black', linestyle='--', label='Vérité')
axs[2,1].plot(temps_np, vx_vrai_2, color='black', linestyle='--', label='Vérité')
axs[3,1].plot(temps_np, vy_vrai_2, color='black', linestyle='--', label='Vérité')

axs[0,2].plot(temps_np, x_vrai_3, color='black', linestyle='--', label='Vérité')
axs[1,2].plot(temps_np, y_vrai_3, color='black', linestyle='--', label='Vérité')
axs[2,2].plot(temps_np, vx_vrai_3, color='black', linestyle='--', label='Vérité')
axs[3,2].plot(temps_np, vy_vrai_3, color='black', linestyle='--', label='Vérité')

axs[0,0].scatter(temps_capteur, x1_mesure, color="red", marker="x", linewidths=0.5)
axs[1,0].scatter(temps_capteur, y1_mesure, color="red", marker="x", linewidths=0.5)
axs[0,1].scatter(temps_capteur, x2_mesure, color="red", marker="x", linewidths=0.5)
axs[1,1].scatter(temps_capteur, y2_mesure, color="red", marker="x", linewidths=0.5)

# 5. Ajustements finaux
# Ajout du label X uniquement sur les graphiques de la dernière ligne
axs[6,0].set_xlabel("Temps (s)")
axs[6,1].set_xlabel("Temps (s)")
axs[6,2].set_xlabel("Temps (s)")

# Ajout d'une seule légende globale pour ne pas surcharger les petits graphiques
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=12)

plt.tight_layout()
plt.subplots_adjust(top=0.92) # Laisse de la place pour la légende globale
fig.suptitle("Analyse des Covariances de l'Error-State EKF (18 dimensions)", fontsize=16, fontweight='bold')
