"""
CHARGEMENT DES MODÈLES ET EXÉCUTION DES FILTRES

Ce fichier regroupe tout ce qui touche aux checkpoints et aux tenseurs
torch. Le reste du code appelle ces fonctions sans jamais manipuler
directement un tenseur de forme (T, 24, 1).

Trois estimateurs sont disponibles pour une même trajectoire :
  - EKF nominal : filtre avec son R par défaut
  - EKF oracle  : filtre avec le vrai R du niveau de bruit testé
  - KalmanNet   : le réseau entraîné

Pourquoi deux EKF ? Voir la section correspondante du README.
"""

import os
import torch

from KalmanNet_Drones import KalmanNetNN, run_knet
from metriques import db_vers_echelle


# ==========================================================================
# 1. CHARGEMENT DES CHECKPOINTS
# ==========================================================================

def chemin_checkpoint(dossier_runs, archi, seed):
    """Reconstruit le chemin d'un checkpoint de baseline.

    Correspond à l'arborescence produite par le script d'entraînement :
    runs/baseline_narrow_archi2_seed42/knet_archi2.pt
    """
    return os.path.join(dossier_runs,
                        f"baseline_narrow_{archi}_seed{seed}",
                        f"knet_{archi}.pt")


def charger_un_modele(sm, chemin, archi_defaut="archi2"):
    """Recharge un KalmanNet entraîné et le met en mode évaluation.

    model.eval() désactive les comportements propres à l'entraînement
    (dropout, batchnorm). Ici ça ne change rien au calcul, mais c'est la
    pratique correcte et ça évite les surprises si l'architecture évolue.
    """
    etat = torch.load(chemin, map_location=sm.device)
    model = KalmanNetNN(sm, archi=etat.get("archi", archi_defaut))
    model.load_state_dict(etat["state_dict"])
    model.eval()
    return model


def charger_les_baselines(sm, dossier_runs, archi, seeds):
    """Charge tous les checkpoints disponibles.

    Renvoie un dictionnaire {seed: modele}. Un checkpoint manquant est
    signalé mais ne fait pas planter : on peut vouloir lancer un balayage
    avec deux graines sur trois.
    """
    modeles = {}
    for seed in seeds:
        chemin = chemin_checkpoint(dossier_runs, archi, seed)
        if os.path.exists(chemin):
            modeles[seed] = charger_un_modele(sm, chemin, archi)
            print(f"   checkpoint chargé : graine {seed}")
        else:
            print(f"!! checkpoint introuvable, graine ignorée : {chemin}")
    return modeles


# ==========================================================================
# 2. EXÉCUTION DES FILTRES
# ==========================================================================

def lancer_ekf_nominal(sm, ekf, Y, U, M):
    """EKF avec son R par défaut, celui du point de fonctionnement nominal."""
    return ekf.run(Y, U, M)[0]


def lancer_ekf_oracle(sm, ekf, Y, U, M, niveau_db):
    """EKF à qui on donne le vrai R du niveau de bruit testé.

    On modifie temporairement sm.R, on lance le filtre, puis on restaure
    la valeur d'origine. Le try/finally garantit que la restauration a
    lieu même si le filtre lève une erreur — sinon tous les calculs
    suivants seraient faussés sans avertissement.
    """
    R_sauvegarde = sm.R
    try:
        sm.R = sm.R_gen * (db_vers_echelle(niveau_db) ** 2)
        return ekf.run(Y, U, M)[0]
    finally:
        sm.R = R_sauvegarde


def lancer_kalmannet(sm, model, Y, U, M):
    """KalmanNet sur une trajectoire. Simple relais vers run_knet."""
    return run_knet(sm, model, Y, U, M)
