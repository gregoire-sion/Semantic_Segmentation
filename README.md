"""
BALAYAGE — AXE 4 : FAMILLE DE COMMANDE

Cet axe est différent des trois autres : il n'est pas NUMÉRIQUE mais
CATÉGORIEL. On ne balaye pas une grandeur continue, on compare des
familles de commande distinctes.

Conséquence : la notion de "franchissement d'un seuil" n'a aucun sens ici
(on n'interpole pas entre "brutal" et "ou"). Ce script réutilise donc la
fonction balayer() du moteur, mais fait sa propre figure en barres et ne
calcule aucune frontière.

EKF NOMINAL comme référence : changer la commande ne modifie pas le R.

À VÉRIFIER AVANT DE LANCER
Les noms de familles ci-dessous viennent de ton generate_dataset.py et de
CFG.TRAIN_CMD_FAMILIES. Si build_command_ood n'accepte pas exactement les
kinds "3phases" et "brutal", ou si les noms de familles diffèrent, il faut
corriger le dictionnaire FAMILLES. Le script s'arrête proprement avec un
message si une famille échoue.

À LANCER :
    python balayage_commande.py
"""

import os
import json
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from KalmanNet_Drones import (CFG, SystemModel, EKF, generate_trajectory,
                              build_command_ood)
from chargement_modeles import charger_les_baselines, lancer_ekf_nominal
from moteur_balayage import balayer


# ==========================================================================
# 1. RÉGLAGES
# ==========================================================================

DOSSIER_RUNS = "./runs"
ARCHI = "archi2"
SEEDS = [42, 1234, 7]

N_TRAJECTOIRES = 150

JEU = "dev"
SEED_DEV = 23250
SEED_TEST = 33250

# Figés au point d'entraînement.
OFFSET_INITIAL = 0.3

DOSSIER_SORTIE = "./runs/balayage_commande"

# La famille vue à l'entraînement, mise en évidence sur la figure.
FAMILLE_ENTRAINEMENT = "nominal_3phases"


# ==========================================================================
# 2. LES FAMILLES DE COMMANDE
# ==========================================================================
# Chaque entrée est une fonction qui prend (rng) et renvoie soit None
# (la commande par défaut de generate_trajectory est alors utilisée),
# soit une séquence de commande construite explicitement.

def _nominal(sm, rng):
    """Commande 3 phases historique, celle de l'entraînement."""
    CFG.TRAIN_CMD_RANDOMIZE = False
    return None


def _phases3_rand(sm, rng):
    """Variante randomisée des 3 phases."""
    CFG.TRAIN_CMD_RANDOMIZE = True
    CFG.TRAIN_CMD_FAMILIES = ("phases3_rand",)
    return None


def _ou(sm, rng):
    """Commande de type processus d'Ornstein-Uhlenbeck."""
    CFG.TRAIN_CMD_RANDOMIZE = True
    CFG.TRAIN_CMD_FAMILIES = ("ou",)
    return None


def _ood_3phases(sm, rng):
    """Famille hors distribution : 3 phases version OOD."""
    CFG.TRAIN_CMD_RANDOMIZE = False
    return build_command_ood(CFG.T, sm.dt, rng, kind="3phases")


def _ood_brutal(sm, rng):
    """Famille hors distribution : manœuvres brutales."""
    CFG.TRAIN_CMD_RANDOMIZE = False
    return build_command_ood(CFG.T, sm.dt, rng, kind="brutal")


FAMILLES = {
    "nominal_3phases": _nominal,
    "phases3_rand": _phases3_rand,
    "ou": _ou,
    "ood_3phases": _ood_3phases,
    "ood_brutal": _ood_brutal,
}


# ==========================================================================
# 3. GÉNÉRATION
# ==========================================================================

def graine_de_base():
    return SEED_DEV if JEU == "dev" else SEED_TEST


def generer_trajectoires(sm, nom_famille, n):
    """n trajectoires pour la famille de commande demandée."""
    fabrique = FAMILLES[nom_famille]
    graine = graine_de_base() + abs(hash(nom_famille)) % 1000
    rng = np.random.default_rng(graine)

    trajectoires = []
    for _ in range(n):
        u_seq = fabrique(sm, rng)
        if u_seq is None:
            trajectoires.append(generate_trajectory(sm, rng, r_scale=1.0))
        else:
            trajectoires.append(
                generate_trajectory(sm, rng, u_seq=u_seq, r_scale=1.0))
    return trajectoires


# ==========================================================================
# 4. FIGURE EN BARRES
# ==========================================================================

def tracer_familles(points, seeds, chemin_figure):
    """Diagramme en barres : une barre par famille et par graine.

    Pas de courbe ici : les familles ne sont pas ordonnées, une ligne
    reliant "ou" à "brutal" n'aurait aucun sens.
    """
    noms = [p["valeur"] for p in points]
    x = np.arange(len(noms))
    largeur = 0.8 / len(seeds)

    fig, (ax_rel, ax_abs) = plt.subplots(1, 2, figsize=(14, 5))

    for i, seed in enumerate(seeds):
        d = [p["par_seed"][str(seed)]["delta_db"] for p in points]
        ic = [p["par_seed"][str(seed)]["ic95"] for p in points]
        ax_rel.bar(x + i * largeur - 0.4 + largeur / 2, d, largeur,
                   yerr=ic, capsize=3, label=f"graine {seed}")

    ax_rel.axhline(0, color="k", lw=1.2)
    ax_rel.axhline(3.0, color="crimson", ls="-.", lw=1.4, label="repère +3 dB")
    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels(noms, rotation=20, ha="right", fontsize=8)
    ax_rel.set_ylabel(r"$\Delta_{dB}$ position   (< 0 : KNet meilleur)")
    ax_rel.set_title("Performance relative")
    ax_rel.grid(True, axis="y", ls=":", alpha=0.7)
    ax_rel.legend(fontsize=8)

    for i, seed in enumerate(seeds):
        mse = [p["par_seed"][str(seed)]["mse_knet_position"] for p in points]
        ax_abs.bar(x + i * largeur - 0.4 + largeur / 2, mse, largeur,
                   label=f"KNet graine {seed}")
    mse_ekf = [p["mse_ekf_position"] for p in points]
    ax_abs.plot(x, mse_ekf, "ks--", lw=2, label="EKF de référence")

    ax_abs.set_yscale("log")
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(noms, rotation=20, ha="right", fontsize=8)
    ax_abs.set_ylabel("MSE position (échelle log)")
    ax_abs.set_title("Performance absolue")
    ax_abs.grid(True, axis="y", ls=":", alpha=0.7, which="both")
    ax_abs.legend(fontsize=8)

    fig.suptitle("Généralisation aux familles de commande — "
                 "baseline étroite archi2")
    fig.tight_layout()
    fig.savefig(chemin_figure, dpi=140)
    plt.close(fig)
    return chemin_figure


# ==========================================================================
# 5. PROGRAMME PRINCIPAL
# ==========================================================================

def main():
    torch.manual_seed(graine_de_base())
    np.random.seed(graine_de_base())

    CFG.INIT_OFFSET_P0 = True
    CFG.INIT_OFFSET_SCALE = OFFSET_INITIAL
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    print("== Balayage axe 4 : famille de commande ==")
    print(f"   jeu d'évaluation : {JEU} (graine de base {graine_de_base()})")
    print(f"   familles         : {list(FAMILLES)}")
    print(f"   entraînement sur : {FAMILLE_ENTRAINEMENT}")
    print(f"   trajectoires     : {N_TRAJECTOIRES} par famille")
    print(f"   EKF de référence : nominal\n")

    sm = SystemModel()
    ekf = EKF(sm)

    modeles = charger_les_baselines(sm, DOSSIER_RUNS, ARCHI, SEEDS)
    if not modeles:
        raise SystemExit("Aucun checkpoint trouvé. Vérifie DOSSIER_RUNS.")
    print()

    debut = time.time()
    points = balayer(
        sm, modeles, list(FAMILLES),
        lambda nom: generer_trajectoires(sm, nom, N_TRAJECTOIRES),
        lambda nom, Y, U, M: lancer_ekf_nominal(sm, ekf, Y, U, M),
    )

    nom_sortie = f"balayage_commande_{JEU}"
    figure = tracer_familles(points, sorted(modeles),
                             os.path.join(DOSSIER_SORTIE, nom_sortie + ".png"))

    sortie = {
        "axe": "famille_de_commande",
        "type": "categoriel",
        "jeu": JEU,
        "seeds": sorted(modeles),
        "familles": list(FAMILLES),
        "famille_entrainement": FAMILLE_ENTRAINEMENT,
        "points": points,
        "duree_s": round(time.time() - debut, 1),
    }
    chemin_json = os.path.join(DOSSIER_SORTIE, nom_sortie + ".json")
    with open(chemin_json, "w", encoding="utf-8") as fh:
        json.dump(sortie, fh, indent=2, ensure_ascii=False)

    print(f"\n== Résultats -> {chemin_json}")
    print(f"== Figure    -> {figure}")


if __name__ == "__main__":
    main()
