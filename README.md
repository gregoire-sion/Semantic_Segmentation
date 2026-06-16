##############################################
# --- SCÉNARIO D'ÉTUDE : COMPENSATION BIAIS ---
##############################################
compenser_biais = False  # Mettre à True pour le filtre parfait, False pour le filtre naïf

if not compenser_biais:
    # 1. On force l'estimation initiale des biais à 0
    X_est[6:8] = 0.0
    X_est[14:16] = 0.0
    X_est[22:24] = 0.0
    
    # 2. On verrouille P0 (Le filtre refuse d'apprendre)
    # (On utilise 1e-8 au lieu de 0.0 pour éviter l'erreur SingularMatrix)
    P_est[6, 6] = P_est[7, 7] = 1e-8
    P_est[14, 14] = P_est[15, 15] = 1e-8
    P_est[22, 22] = P_est[23, 23] = 1e-8
    
    # 3. On verrouille Q (Le filtre refuse de douter)
    Q_kalman[6, 6] = Q_kalman[7, 7] = 1e-8
    Q_kalman[14, 14] = Q_kalman[15, 15] = 1e-8
    Q_kalman[22, 22] = Q_kalman[23, 23] = 1e-8
##############################################
