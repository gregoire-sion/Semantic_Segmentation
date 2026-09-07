"""
BALAYAGE — AXE 2 : AMPLITUDE DE L'OFFSET INITIAL

L'offset initial est l'écart entre l'état réel du système au départ et
l'état que le filtre suppose au départ. La trajectoire vraie démarre à
x0 + echelle * (L @ xi), où L vient de la décomposition de Cholesky de
P0 ; le filtre, lui, démarre toujours à x0 exactement.

echelle = 0.3 est le point d'entraînement de la baseline étroite.
echelle = 1.0 correspond à une perturbation cohérente avec P0.

DIFFÉRENCE AVEC L'AXE BRUIT : ici on utilise l'EKF NOMINAL, pas l'oracle.
Changer l'offset initial ne modifie en rien le R du filtre, donc son
réglage de bruit reste correct. Un EKF oracle n'aurait aucun sens ici.

À LANCER :
    python balayage_offset.py

Essai à blanc : N_TRAJECTOIRES = 10 et ECHELLES = [0.3, 1.0, 3.0]
"""

import numpy as np
import torch

from KalmanNet_Drones import CFG, SystemModel, EKF, generate_trajectory
from chargement_modeles import charger_les_baselines, lancer_ekf_nominal
from moteur_balayage import lancer_un_axe


# ==========================================================================
# 1. RÉGLAGES
# ==========================================================================

DOSSIER_RUNS = "./runs"
ARCHI = "archi2"
SEEDS = [42, 1234, 7]

# Amplitudes balayées. 0.3 = point d'entraînement, 0.0 = le filtre part
# de l'état vrai (cas irréaliste mais utile comme borne).
ECHELLES = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0]

N_TRAJECTOIRES = 150

JEU = "dev"                 # "test" uniquement à l'étape 5
SEED_DEV = 21250
SEED_TEST = 31250

# Figés au point d'entraînement : on ne balaye qu'un facteur à la fois.
NIVEAU_BRUIT_DB = 0.0       # bruit nominal
COMMANDE_RANDOMISEE = False

DOSSIER_SORTIE = "./runs/balayage_offset"


# ==========================================================================
# 2. CE QUI EST PROPRE À L'AXE OFFSET
# ==========================================================================

def graine_de_base():
    return SEED_DEV if JEU == "dev" else SEED_TEST


def generer_trajectoires(sm, echelle, n):
    """n trajectoires avec l'amplitude d'offset initial demandée.

    generate_trajectory lit CFG.INIT_OFFSET_SCALE au moment de l'appel :
    il suffit donc de modifier ce réglage avant de générer. C'est le même
    mécanisme que dans le script de baseline.

    La graine dépend de l'échelle mais pas du modèle : les trois
    baselines voient exactement les mêmes trajectoires.
    """
    CFG.INIT_OFFSET_SCALE = echelle
    CFG.INIT_OFFSET_P0 = (echelle > 0.0)

    graine = graine_de_base() + int(round(echelle * 100))
    rng = np.random.default_rng(graine)
    return [generate_trajectory(sm, rng, r_scale=1.0) for _ in range(n)]


# ==========================================================================
# 3. PROGRAMME PRINCIPAL
# ==========================================================================

def main():
    torch.manual_seed(graine_de_base())
    np.random.seed(graine_de_base())

    CFG.TRAIN_CMD_RANDOMIZE = COMMANDE_RANDOMISEE

    print("== Balayage axe 2 : amplitude de l'offset initial ==")
    print(f"   jeu d'évaluation : {JEU} (graine de base {graine_de_base()})")
    print(f"   échelles         : {ECHELLES}")
    print(f"   trajectoires     : {N_TRAJECTOIRES} par échelle")
    print(f"   bruit figé       : {NIVEAU_BRUIT_DB} dB (nominal)")
    print(f"   EKF de référence : nominal (pas d'oracle sur cet axe)\n")

    sm = SystemModel()
    ekf = EKF(sm)

    modeles = charger_les_baselines(sm, DOSSIER_RUNS, ARCHI, SEEDS)
    if not modeles:
        raise SystemExit("Aucun checkpoint trouvé. Vérifie DOSSIER_RUNS.")
    print()

    config = {
        "nom": f"balayage_offset_{JEU}",
        "jeu": JEU,
        "dossier_sortie": DOSSIER_SORTIE,
        "valeurs": ECHELLES,
        "generer_trajectoires":
            lambda echelle: generer_trajectoires(sm, echelle, N_TRAJECTOIRES),
        "estimer_ekf":
            lambda echelle, Y, U, M: lancer_ekf_nominal(sm, ekf, Y, U, M),
        "titre": "Généralisation à l'offset initial — baseline étroite archi2",
        "label_x": "Amplitude de l'offset initial (x écart-type de P0)",
        "valeur_entrainement": 0.3,
        "seuils": (0.0, 3.0),
        "seuil_repere": 3.0,
    }
    lancer_un_axe(sm, modeles, config)


if __name__ == "__main__":
    main()
