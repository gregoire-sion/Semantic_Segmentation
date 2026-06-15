    #----Choix de la commande dynamique----
    
    # PHASE 1 : Le "Zig-Zag" d'Initialisation (0 à 4 secondes)
    # Objectif : Casser l'inobservabilité initiale et résoudre le "Triangle sur Pivot"
    # Action : Les drones accélèrent latéralement dans des directions différentes.
    if step < ((t_max/dt) / 4):
        # On donne une impulsion asymétrique
        Ax1, Ay1 = 1.0,  0.5
        Ax2, Ay2 = 1.0, -0.5  # Le drone 2 s'écarte vers le bas
        Ax3, Ay3 = 1.0,  0.8  # Le drone 3 s'écarte vers le haut
        
        u_vrai = np.array([Ax1, Ay1, Ax2, Ay2, Ax3, Ay3])
        u_kalman = u_vrai.copy()

    # PHASE 2 : La Respiration Déphasée (4 à 12 secondes)
    # Objectif : Maintenir l'observabilité maximale pendant les manœuvres
    # Action : Les drones tournent, mais avec des fréquences légèrement différentes
    elif step >= ((t_max/dt) / 4) and step < (3 * (t_max/dt) / 4):
        # Fréquences déphasées pour déformer continuellement le triangle
        omega_1 = 2.0  # Le Maître tourne doucement
        omega_2 = 2.5  # Le Drone 2 tourne un peu plus vite
        omega_3 = 1.5  # Le Drone 3 tourne plus lentement

        phi_x1 = omega_1 * t
        phi_y1 = omega_1 * t
        phi_x2 = omega_2 * t
        phi_y2 = omega_2 * t
        phi_x3 = omega_3 * t
        phi_y3 = omega_3 * t

        u_vrai = np.array([
            1.0 * np.cos(phi_x1), 1.0 * np.sin(phi_y1),
            1.0 * np.cos(phi_x2), 1.0 * np.sin(phi_y2),
            1.0 * np.cos(phi_x3), 1.0 * np.sin(phi_y3)
        ])
        u_kalman = u_vrai.copy()

    # PHASE 3 : Le Retour au Calme (12 à 16 secondes)
    # Objectif : Prouver que l'essaim a mémorisé sa géométrie
    # Action : Retour à une ligne droite lisse. L'incertitude va doucement remonter, mais sans retard.
    else:
        u_vrai = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        u_kalman = u_vrai.copy()
