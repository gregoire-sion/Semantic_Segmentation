import matplotlib.pyplot as plt
import numpy as np

# 1. Conversion des historiques
temps_np = np.array(temps)[:traj_kalman.shape[0]]
P_hist_np = np.array(P_historique)[:traj_kalman.shape[0]]

# 2. Liste des noms pour que tes 18 graphiques soient lisibles
noms_etats = [
    "X Drone 1", "Y Drone 1", "Vx Drone 1", "Vy Drone 1",
    "X Drone 2 (Dérive)", "Y Drone 2 (Dérive)", "Vx Drone 2", "Vy Drone 2",
    "X Drone 3", "Y Drone 3", "Vx Drone 3", "Vy Drone 3",
    "Biais Pos X", "Biais Pos Y",
    "Biais Vit X", "Biais Vit Y",
    "Biais Acc X", "Biais Acc Y"
]

# 3. Création de la figure géante (6 lignes, 3 colonnes)
fig, axs = plt.subplots(6, 3, figsize=(18, 20), sharex=True)
axs = axs.flatten() # Transforme la grille 2D en une simple liste 1D pour la boucle

# 4. Boucle sur les 18 variables d'état
for i in range(18):
    # Extraction de l'état estimé par l'EKF
    etat_estime = traj_kalman[:, 0, i] 
    
    # Extraction de la variance (diagonale de P) et calcul de l'écart-type (sigma)
    variance = P_hist_np[:, i, i]
    sigma = np.sqrt(variance)
    
    # Calcul des bornes du couloir de confiance à 99.7% (+/- 3 sigma)
    borne_haute = etat_estime + 3 * sigma
    borne_basse = etat_estime - 3 * sigma
    
    # Tracé de la variable d'état (ligne pleine)
    axs[i].plot(temps_np, etat_estime, color='blue', label='Estimation EKF')
    
    # Tracé du couloir de covariance (zone ombrée)
    axs[i].fill_between(temps_np, borne_basse, borne_haute, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    
    # Esthétique du graphique
    axs[i].set_title(noms_etats[i], fontsize=10, fontweight='bold')
    axs[i].grid(True, linestyle=':', alpha=0.7)
    
    # (Optionnel) Si tu as stocké la vérité terrain dans une matrice "traj_vraie", 
    # tu peux la rajouter ici pour voir si elle reste bien dans le couloir !
    # axs[i].plot(temps_np, traj_vraie[:, i], color='black', linestyle='--', label='Vérité')

# 5. Ajustements finaux
# Ajout du label X uniquement sur les graphiques de la dernière ligne
for ax in axs[-3:]:
    ax.set_xlabel("Temps (s)")

# Ajout d'une seule légende globale pour ne pas surcharger les petits graphiques
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=12)

plt.tight_layout()
plt.subplots_adjust(top=0.92) # Laisse de la place pour la légende globale
fig.suptitle("Analyse des Covariances de l'Error-State EKF (18 dimensions)", fontsize=16, fontweight='bold')
plt.show()



Bonjour,

Je ne sais pas comment l’écrire mais je sais que je dois le dire. Je préfère donc le faire de la façon la plus ouverte et me concentrer sur des faits.

Après discussion avec Grégoire sur son alternance plusieurs points m’étonnent :
•	Il est chez nous depuis 2024 (même si une alternance ce n’est qu’à temps partiel cela reste un temps long)
•	Il n’a pas commencé à écrire de rapport, je me serai attendu que les travaux des années passées, même si la piste n’a pas été fructueuse, soient archivés proprement sous forme de sections dans un rapport. Quelles traces a-t-on des travaux passés ? Cela permet aussi d’éviter le mur du rapport final, et indique pour moi un malaise sur l’ensemble des travaux, pour lesquels peu de choses semblent avoir abouties.
•	Aucune présentation, technique ou autre, n’a été faite par Grégoire avec quelqu’un d’autre de Stéphane. La culture de l’échange est au cœur des valeurs de GCN, et il est dommage qu’il n’est pas eu l’occasion de mettre en valeurs ses travaux. Et si l’absence de présentation des résultats de Grégoire était due à un manque de résultats,  et bien cette présentation aurait été l’occasion de discuter en groupe des problèmes du stage.

J’encourage Grégoire à faire dans le mois qui vient une présentation auprès des personnes intéressées sur ces travaux et leurs enjeux. Il ne s’agit pas d’une soutenance, mais d’un point d’échange avec d’autres acteurs de GCN, pour valider la direction du travail, les enjeux et les outils. Ce point demande un support de présentation type power-point, mais qui sert avant tout à animer une discussion et n’a donc pas besoin d’une maturité aussi élevée qu’une soutenance.

Merci

Bonjour Corentin,
Merci pour ton mail et tes interrogations ;
Je pense que le mieux est de discuter ensemble oralement plutôt que d’écrire un long mail et d’encombrer des boites mails déjà bien remplies.
Cela fait longtemps que l’on ne s’est pas croisé.
J’apprécie toujours autant de croiser des avis et points de vue avec des collègues GCN et MD de façon générale.
J’ai d’ailleurs souvent l’occasion de discuter avec Jérémy, les François, Bénédicte, Rodolphe, Clara … les senseurs inertielles …
En ce qui concerne Grégoire et l’ENSTA, il y a peut-être une information que tu n’as pas.
L’ENSTA a fait évoluer sa formule d’apprentissage.
Les apprentis ne sont maintenant plus nécessairement en filière systèmes complexes.
Les apprentis ont maintenant également accès aux autres filières.
Grégoire a choisi de se réorienter en filière IA.
GCN a soutenu Grégoire dans son choix.
C’est un choix pas facile car il y a plusieurs conséquences.
Le rythme de l’alternance et le sujet de l’alternance de Grégoire ont été modifiés pour mieux correspondre à sa nouvelle filière.
C’est un choix courageux de Grégoire car le nouveau rythme d’alternance entre cours et périodes en entreprise a été très soutenu en 2025.
C’est également un choix qui demande à Grégoire d’apprendre et de mettre en œuvre des techniques pas nécessairement évidentes liées à l’IA et à l’apprentissage.
Ce nouveau rythme d’alternance fait que Grégoire a moins travaillé en entreprise en 2025 et sera plus avec nous à partir de maintenant.
Le travail de Grégoire en entreprise sur son nouveau sujet commence principalement maintenant.
Cela explique pourquoi Grégoire présentera ces travaux davantage selon un rythme de stage de fin d’école (soutenance intermédiaire et finale).
J’espère que ces explications répondent à tes interrogations.
Bonne journée,
Cordialement,
Stéphane.

