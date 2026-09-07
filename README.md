"""
MÉTRIQUES COMMUNES

Ce fichier définit la façon de mesurer, et rien d'autre : pas de
génération de données, pas d'entraînement, pas de figure. Il est importé
par tous les scripts de balayage, pour qu'ils mesurent exactement la même
chose avec le même code.

IMPORTANT : le dictionnaire GROUPES ci-dessous doit être identique à
celui de ton script de baseline (Train_archi2_etroit.py). Si tu modifies
l'un, modifie l'autre — sinon la baseline et les balayages ne mesureraient
plus la même grandeur, et les comparaisons n'auraient plus de sens.
"""

import numpy as np


# ==========================================================================
# 1. DÉCOUPAGE DU VECTEUR D'ÉTAT
# ==========================================================================
# Le vecteur d'état fait 24 composantes = 3 drones x 8 variables.
# Pour chaque drone : x, y, vx, vy, ax, ay, bx, by
# Donc le drone 1 occupe les indices 0 à 7, le drone 2 les indices 8 à 15,
# le drone 3 les indices 16 à 23.

BASES = (0, 8, 16)          # indice de départ de chaque drone

GROUPES = {
    "position":     [0, 1,  8,  9, 16, 17],
    "vitesse":      [2, 3, 10, 11, 18, 19],
    "acceleration": [4, 5, 12, 13, 20, 21],
    "biais":        [6, 7, 14, 15, 22, 23],
}


# ==========================================================================
# 2. ERREUR QUADRATIQUE
# ==========================================================================

def mse_groupe(xhat, xtrue, indices):
    """Erreur quadratique moyenne sur un sous-ensemble de composantes.

    Le .mean() sans argument moyenne TOUT d'un coup : les composantes
    choisies ET les pas de temps de la trajectoire. Le résultat est donc
    un seul nombre par trajectoire.
    """
    return ((xhat[:, indices, 0] - xtrue[:, indices, 0]) ** 2).mean().item()


def mse_par_groupe(xhat, xtrue):
    """Dictionnaire {nom du groupe : MSE} pour une trajectoire."""
    return {nom: mse_groupe(xhat, xtrue, idx) for nom, idx in GROUPES.items()}


# ==========================================================================
# 3. MÉTRIQUE CENTRALE DE L'ÉTUDE
# ==========================================================================

def delta_db(mse_knet, mse_ekf):
    """Delta_dB = 10 log10(MSE_KalmanNet / MSE_EKF).

    Négatif  -> KalmanNet fait mieux que l'EKF
    Nul      -> les deux se valent (point de parité)
    Positif  -> KalmanNet fait moins bien

    Pour repasser en facteur linéaire : facteur = 10^(Delta_dB / 10).
    Exemple : -5.7 dB  ->  0.27, soit une erreur 3.7 fois plus faible.
    """
    return 10.0 * np.log10(mse_knet / mse_ekf)


def moyenne_ic95(valeurs):
    """Moyenne et demi-largeur de l'intervalle de confiance à 95 %.

    IC95 = 1.96 * ecart_type / racine(n)

    ddof=1 est le diviseur (n-1), estimateur sans biais de l'écart-type.
    Le 1.96 vient de l'approximation normale ; pour n >= 50 c'est
    suffisant (la valeur exacte de Student vaut 2.01 à n=50).
    """
    v = np.asarray(valeurs, dtype=float)
    n = len(v)
    if n < 2:
        return float(v.mean()), float("nan")
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(n))


# ==========================================================================
# 4. CONVERSION DES NIVEAUX DE BRUIT
# ==========================================================================

def db_vers_echelle(niveau_db):
    """Convertit un niveau de bruit en dB vers un facteur multiplicatif.

    Convention reprise de ton code : r_scale = 10^(-niveau_db / 20).

    niveau_db =   0  ->  r_scale = 1     bruit nominal (celui de l'entraînement)
    niveau_db = -20  ->  r_scale = 10    dix fois plus bruité
    niveau_db = +20  ->  r_scale = 0.1   dix fois moins bruité

    Attention au signe : un niveau en dB PLUS GRAND veut dire MOINS de bruit.
    """
    return 10.0 ** (-niveau_db / 20.0)


# ==========================================================================
# 5. AFFICHAGE
# ==========================================================================

def afficher_tableau(mse_knet, mse_ekf):
    """Tableau MSE par groupe d'états, avec le gain en dB et en facteur."""
    print(f"{'groupe':<14} {'MSE KNet':>12} {'MSE EKF':>12} "
          f"{'gain dB':>10} {'facteur':>10}")
    for nom in GROUPES:
        mk, me = mse_knet[nom], mse_ekf[nom]
        d = delta_db(mk, me)
        facteur = 10.0 ** (-d / 10.0)
        print(f"{nom:<14} {mk:>12.4f} {me:>12.4f} {d:>+10.2f} "
              f"{facteur:>9.2f}x")
