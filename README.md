"""
BALAYAGE — AXE 1 : NIVEAU DE BRUIT DE MESURE

Ce script ne contient QUE ce qui est propre à l'axe "bruit" :
  - les niveaux de bruit à balayer
  - comment générer des trajectoires à un niveau donné
  - quel EKF sert de référence (ici l'oracle, voir README)

Tout le travail commun est délégué à moteur_balayage.py.

Les modèles évalués sont les trois baselines ÉTROITES déjà entraînées.
Aucun réentraînement n'a lieu ici.

À LANCER :
    python balayage_bruit.py

Essai à blanc (quelques minutes) : mettre en bas de la section réglages
    N_TRAJECTOIRES = 10
    NIVEAUX_DB = [-10, 0, 10]
"""

import numpy as np
import torch

from KalmanNet_Drones import CFG, SystemModel, EKF, generate_trajectory
from metriques import db_vers_echelle
from chargement_modeles import charger_les_baselines, lancer_ekf_oracle
from moteur_balayage import lancer_un_axe


# ==========================================================================
# 1. RÉGLAGES
# ==========================================================================

DOSSIER_RUNS = "./runs"
ARCHI = "archi2"
SEEDS = [42, 1234, 7]

# Modèles évalués : "narrow" = baseline étroite, "wide" = modèle
# entraîné avec randomisation de domaine. Les sorties sont écrites
# dans des dossiers distincts, donc rien n'est écrasé.
VARIANTE = "narrow"

# Niveaux de bruit balayés. Rappel du signe : plus le nombre est GRAND,
# MOINS il y a de bruit. 0 dB = le niveau vu à l'entraînement.
NIVEAUX_DB = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 30]

N_TRAJECTOIRES = 150        # par niveau ; 150 donne un IC95 d'environ 1 dB

# Jeu d'évaluation : "dev" pour explorer et diagnostiquer (étapes 1 à 4),
# "test" UNIQUEMENT pour la mesure finale de l'étape 5. Ne pas regarder
# le jeu "test" avant d'avoir figé la correction, sinon on retombe dans
# la circularité que le protocole cherche à éviter.
JEU = "dev"

SEED_DEV = 20250
SEED_TEST = 30250

# Ces deux réglages sont FIGÉS au point d'entraînement : on ne balaye
# qu'un seul facteur à la fois.
OFFSET_INITIAL = 0.3
COMMANDE_RANDOMISEE = False

DOSSIER_SORTIE = f"./runs/balayage_bruit_{VARIANTE}"


# ==========================================================================
# 2. CE QUI EST PROPRE À L'AXE BRUIT
# ==========================================================================

def graine_de_base():
    """Graine du jeu d'évaluation choisi."""
    return SEED_DEV if JEU == "dev" else SEED_TEST


def generer_trajectoires(sm, niveau_db, n):
    """n trajectoires générées au niveau de bruit demandé.

    La graine ne dépend que du niveau et du jeu, jamais du modèle : les
    trois baselines sont donc évaluées sur exactement les mêmes
    trajectoires (comparaison appariée).

    Les jeux dev et test utilisent des graines très éloignées, donc des
    trajectoires entièrement différentes.
    """
    graine = graine_de_base() + int(round(niveau_db))
    rng = np.random.default_rng(graine)
    r_scale = db_vers_echelle(niveau_db)
    return [generate_trajectory(sm, rng, r_scale=r_scale) for _ in range(n)]


# ==========================================================================
# 3. PROGRAMME PRINCIPAL
# ==========================================================================

def main():
    torch.manual_seed(graine_de_base())
    np.random.seed(graine_de_base())

    # On fige tous les facteurs sauf celui qu'on balaye.
    CFG.TRAIN_CMD_RANDOMIZE = COMMANDE_RANDOMISEE
    CFG.INIT_OFFSET_P0 = True
    CFG.INIT_OFFSET_SCALE = OFFSET_INITIAL

    print("== Balayage axe 1 : niveau de bruit de mesure ==")
    print(f"   jeu d'évaluation : {JEU} (graine de base {graine_de_base()})")
    print(f"   niveaux          : {NIVEAUX_DB}")
    print(f"   trajectoires     : {N_TRAJECTOIRES} par niveau")
    print(f"   offset figé      : {OFFSET_INITIAL}")
    print(f"   modèles          : variante {VARIANTE}, aucun réentraînement\n")

    sm = SystemModel()
    ekf = EKF(sm)

    modeles = charger_les_baselines(sm, DOSSIER_RUNS, ARCHI, SEEDS,
                                 VARIANTE)
    if not modeles:
        raise SystemExit("Aucun checkpoint trouvé. Vérifie DOSSIER_RUNS.")
    print()

    # Les deux fonctions ci-dessous adaptent les fonctions de cet axe à la
    # signature attendue par le moteur : il appelle generer_trajectoires
    # avec la seule valeur balayée, et estimer_ekf avec la valeur et une
    # trajectoire.
    config = {
        "nom": f"balayage_bruit_{VARIANTE}_{JEU}",
        "jeu": JEU,
        "dossier_sortie": DOSSIER_SORTIE,
        "valeurs": NIVEAUX_DB,
        "generer_trajectoires":
            lambda niveau: generer_trajectoires(sm, niveau, N_TRAJECTOIRES),
        "estimer_ekf":
            lambda niveau, Y, U, M: lancer_ekf_oracle(sm, ekf, Y, U, M, niveau),
        "titre": "Généralisation au niveau de bruit — baseline étroite archi2",
        "label_x": "Niveau de bruit  1/r²  [dB]   (gauche = plus bruité)",
        "valeur_entrainement": 0,
        "seuils": (0.0, 3.0),
        "seuil_repere": 3.0,
    }
    lancer_un_axe(sm, modeles, config)


if __name__ == "__main__":
    main()
