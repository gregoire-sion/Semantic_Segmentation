"""
COMPARAISON DES BALAYAGES : BASELINE ÉTROITE vs MODÈLE LARGE

Ce script ne lance aucun calcul. Il relit les fichiers JSON déjà produits
par les balayages et les met face à face, axe par axe.

CE QU'IL PRODUIT
  - une figure par axe, superposant narrow et wide
  - un tableau récapitulatif : frontières et Delta_dB au point nominal
  - un JSON de synthèse

CE QU'IL FAUT AVOIR LANCÉ AVANT
  balayage_offset.py, balayage_bruit.py, balayage_commande.py
  une fois avec VARIANTE = "narrow", une fois avec VARIANTE = "wide"

Le script cherche les JSON dans les dossiers runs/balayage_<axe>_<variante>/
et prend le premier trouvé. Un axe absent est signalé et ignoré : on peut
donc lancer la comparaison avec seulement deux axes sur trois.


UNE PRÉCAUTION DE LECTURE
Les deux modèles sont testés dans des conditions IDENTIQUES, celles de la
baseline étroite (offset figé à 0,3 sur l'axe bruit, bruit figé à 0 dB sur
l'axe offset). C'est ce qu'il faut pour une comparaison équitable, mais
cela signifie que le modèle large n'est testé à son propre optimum sur
aucun axe croisé. Un minimum décalé n'est donc pas une anomalie.

À LANCER :
    python comparaison_narrow_wide.py
"""

import os
import glob
import json

import numpy as np
import matplotlib.pyplot as plt

from moteur_balayage import franchissement


# ==========================================================================
# 1. RÉGLAGES
# ==========================================================================

DOSSIER_RUNS = "./runs"
JEU = "dev"

# Axes numériques : valeur balayée continue, courbe et frontière possibles.
AXES_NUMERIQUES = {
    "balayage_offset": {
        "titre": "Offset initial",
        "label_x": "Amplitude de l'offset initial (x écart-type de P0)",
        "point_entrainement_narrow": 0.3,
        "point_entrainement_wide": 1.0,
    },
    "balayage_bruit": {
        "titre": "Niveau de bruit de mesure",
        "label_x": "Niveau de bruit  1/r²  [dB]   (gauche = plus bruité)",
        "point_entrainement_narrow": 0.0,
        "point_entrainement_wide": None,   # plage [-10, +30], pas un point
    },
}

# Axe catégoriel : familles non ordonnées, diagramme en barres, pas de
# frontière (on n'interpole pas entre deux familles).
AXES_CATEGORIELS = {
    "balayage_commande": {"titre": "Familles de commande"},
}

SEUILS = (0.0, 3.0)

DOSSIER_SORTIE = "./runs/comparaison"

COULEURS = {"narrow": "tab:blue", "wide": "tab:red"}
ETIQUETTES = {"narrow": "baseline étroite", "wide": "modèle large"}


# ==========================================================================
# 2. LECTURE DES FICHIERS
# ==========================================================================

def charger_json(nom_axe, variante):
    """Trouve et lit le JSON d'un balayage.

    On cherche par motif plutôt que par nom exact : les fichiers produits
    avant l'ajout de la constante VARIANTE s'appellent simplement
    balayage_bruit_dev.json, les plus récents balayage_bruit_narrow_dev.json.
    Le dossier, lui, porte toujours la variante.
    """
    dossier = os.path.join(DOSSIER_RUNS, f"{nom_axe}_{variante}")
    fichiers = sorted(glob.glob(os.path.join(dossier, "*.json")))
    if not fichiers:
        print(f"!! aucun JSON dans {dossier}")
        return None
    with open(fichiers[0], encoding="utf-8") as fh:
        return json.load(fh)


def courbe_moyenne(donnees):
    """Delta_dB moyenné sur les graines, valeur par valeur.

    On recalcule depuis les points plutôt que de lire la clé "resume" :
    cela marche pour les deux formats de JSON (numérique et catégoriel).
    """
    seeds = [str(s) for s in donnees["seeds"]]
    valeurs, moyennes, incertitudes = [], [], []
    for point in donnees["points"]:
        valeurs.append(point["valeur"])
        moyennes.append(np.mean([point["par_seed"][s]["delta_db"]
                                 for s in seeds]))
        incertitudes.append(np.mean([point["par_seed"][s]["ic95"]
                                     for s in seeds]))
    return valeurs, np.array(moyennes), np.array(incertitudes)


def mse_moyennes(donnees):
    """MSE position : celle de KalmanNet (moyennée sur les graines) et
    celle de l'EKF de référence."""
    seeds = [str(s) for s in donnees["seeds"]]
    knet, ekf = [], []
    for point in donnees["points"]:
        knet.append(np.mean([point["par_seed"][s]["mse_knet_position"]
                             for s in seeds]))
        ekf.append(point["mse_ekf_position"])
    return np.array(knet), np.array(ekf)


# ==========================================================================
# 3. FIGURE POUR UN AXE NUMÉRIQUE
# ==========================================================================

def figure_axe_numerique(nom_axe, config, donnees_par_variante, chemin):
    """Deux panneaux, deux variantes superposées."""
    fig, (ax_rel, ax_abs) = plt.subplots(1, 2, figsize=(13, 5))

    for variante, donnees in donnees_par_variante.items():
        valeurs, moyennes, ic = courbe_moyenne(donnees)
        couleur = COULEURS[variante]

        ax_rel.plot(valeurs, moyennes, marker="o", lw=2, color=couleur,
                    label=ETIQUETTES[variante])
        ax_rel.fill_between(valeurs, moyennes - ic, moyennes + ic,
                            color=couleur, alpha=0.15)

        knet, ekf = mse_moyennes(donnees)
        ax_abs.plot(valeurs, knet, marker="o", lw=2, color=couleur,
                    label=f"KNet {ETIQUETTES[variante]}")

    # L'EKF est le même dans les deux cas : on ne le trace qu'une fois.
    une_variante = next(iter(donnees_par_variante.values()))
    valeurs, _, _ = courbe_moyenne(une_variante)
    _, ekf = mse_moyennes(une_variante)
    ax_abs.plot(valeurs, ekf, "ks--", lw=2, label="EKF de référence")

    ax_rel.axhline(0, color="k", lw=1.2, label="parité avec l'EKF")
    ax_rel.axhline(3.0, color="crimson", ls="-.", lw=1.2, label="repère +3 dB")

    for variante in ("narrow", "wide"):
        point = config.get(f"point_entrainement_{variante}")
        if point is not None and variante in donnees_par_variante:
            ax_rel.axvline(point, color=COULEURS[variante], ls=":", lw=1.4)

    ax_rel.set_xlabel(config["label_x"])
    ax_rel.set_ylabel(r"$\Delta_{dB}$ position   (< 0 : KNet meilleur)")
    ax_rel.set_title("Performance relative")
    ax_rel.grid(True, ls=":", alpha=0.7)
    ax_rel.legend(fontsize=8)

    ax_abs.set_yscale("log")
    ax_abs.set_xlabel(config["label_x"])
    ax_abs.set_ylabel("MSE position (échelle log)")
    ax_abs.set_title("Performance absolue")
    ax_abs.grid(True, ls=":", alpha=0.7, which="both")
    ax_abs.legend(fontsize=8)

    fig.suptitle(f"{config['titre']} — étroite vs large "
                 f"(pointillés : point d'entraînement de chaque modèle)")
    fig.tight_layout()
    fig.savefig(chemin, dpi=140)
    plt.close(fig)
    return chemin


# ==========================================================================
# 4. FIGURE POUR UN AXE CATÉGORIEL
# ==========================================================================

def figure_axe_categoriel(config, donnees_par_variante, chemin):
    """Barres groupées : une paire narrow/wide par famille."""
    une_variante = next(iter(donnees_par_variante.values()))
    noms = [p["valeur"] for p in une_variante["points"]]
    x = np.arange(len(noms))
    largeur = 0.35

    fig, (ax_rel, ax_abs) = plt.subplots(1, 2, figsize=(14, 5))

    for i, (variante, donnees) in enumerate(donnees_par_variante.items()):
        _, moyennes, ic = courbe_moyenne(donnees)
        decalage = (i - 0.5) * largeur
        ax_rel.bar(x + decalage, moyennes, largeur, yerr=ic, capsize=3,
                   color=COULEURS[variante], label=ETIQUETTES[variante])

        knet, _ = mse_moyennes(donnees)
        ax_abs.bar(x + decalage, knet, largeur, color=COULEURS[variante],
                   label=f"KNet {ETIQUETTES[variante]}")

    _, ekf = mse_moyennes(une_variante)
    ax_abs.plot(x, ekf, "ks--", lw=2, label="EKF de référence")

    ax_rel.axhline(0, color="k", lw=1.2)
    ax_rel.axhline(3.0, color="crimson", ls="-.", lw=1.2, label="repère +3 dB")
    for ax in (ax_rel, ax_abs):
        ax.set_xticks(x)
        ax.set_xticklabels(noms, rotation=20, ha="right", fontsize=8)
        ax.grid(True, axis="y", ls=":", alpha=0.7)
        ax.legend(fontsize=8)

    ax_rel.set_ylabel(r"$\Delta_{dB}$ position")
    ax_rel.set_title("Performance relative")
    ax_abs.set_yscale("log")
    ax_abs.set_ylabel("MSE position (échelle log)")
    ax_abs.set_title("Performance absolue")

    fig.suptitle(f"{config['titre']} — étroite vs large")
    fig.tight_layout()
    fig.savefig(chemin, dpi=140)
    plt.close(fig)
    return chemin


# ==========================================================================
# 5. SYNTHÈSE CHIFFRÉE
# ==========================================================================

def frontieres(donnees, seuils):
    """Franchissement de chaque seuil sur la courbe moyenne."""
    valeurs, moyennes, _ = courbe_moyenne(donnees)
    return {str(s): franchissement(valeurs, list(moyennes), s) for s in seuils}


def texte_frontiere(valeur):
    return f"{valeur:+.2f}" if valeur is not None else "hors plage"


def synthese_numerique(nom_axe, donnees_par_variante):
    """Tableau des frontières, et écart entre les deux variantes."""
    ligne = {}
    for variante, donnees in donnees_par_variante.items():
        ligne[variante] = frontieres(donnees, SEUILS)

    print(f"\n-- {nom_axe} --")
    print(f"{'seuil':>10} {'étroite':>14} {'large':>14} {'écart':>14}")
    for seuil in SEUILS:
        cle = str(seuil)
        n = ligne.get("narrow", {}).get(cle)
        w = ligne.get("wide", {}).get(cle)
        if n is not None and w is not None:
            ecart = f"{w - n:+.2f}"
        else:
            ecart = "n/a"
        etiquette = "parité" if seuil == 0 else f"+{seuil:g} dB"
        print(f"{etiquette:>10} {texte_frontiere(n):>14} "
              f"{texte_frontiere(w):>14} {ecart:>14}")
    return ligne


def synthese_categorielle(nom_axe, donnees_par_variante):
    """Delta_dB par famille, pour les deux variantes."""
    une = next(iter(donnees_par_variante.values()))
    noms = [p["valeur"] for p in une["points"]]

    resultats = {}
    for variante, donnees in donnees_par_variante.items():
        _, moyennes, _ = courbe_moyenne(donnees)
        resultats[variante] = {n: float(m) for n, m in zip(noms, moyennes)}

    print(f"\n-- {nom_axe} --")
    print(f"{'famille':<24} {'étroite':>10} {'large':>10} {'gain':>10}")
    for nom in noms:
        n = resultats.get("narrow", {}).get(nom)
        w = resultats.get("wide", {}).get(nom)
        gain = f"{n - w:+.2f}" if (n is not None and w is not None) else "n/a"
        tn = f"{n:+.2f}" if n is not None else "  -"
        tw = f"{w:+.2f}" if w is not None else "  -"
        print(f"{nom:<24} {tn:>10} {tw:>10} {gain:>10}")
    return resultats


# ==========================================================================
# 6. PROGRAMME PRINCIPAL
# ==========================================================================

def main():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    print("== Comparaison baseline étroite vs modèle large ==")
    print("   (les deux modèles sont testés dans des conditions identiques,")
    print("    celles de la baseline étroite)\n")

    synthese = {}
    figures = []

    # --- Axes numériques -------------------------------------------------
    for nom_axe, config in AXES_NUMERIQUES.items():
        donnees = {}
        for variante in ("narrow", "wide"):
            d = charger_json(nom_axe, variante)
            if d is not None:
                donnees[variante] = d
        if not donnees:
            continue

        chemin = os.path.join(DOSSIER_SORTIE, f"comparaison_{nom_axe}.png")
        figures.append(figure_axe_numerique(nom_axe, config, donnees, chemin))
        synthese[nom_axe] = synthese_numerique(nom_axe, donnees)

    # --- Axe catégoriel --------------------------------------------------
    for nom_axe, config in AXES_CATEGORIELS.items():
        donnees = {}
        for variante in ("narrow", "wide"):
            d = charger_json(nom_axe, variante)
            if d is not None:
                donnees[variante] = d
        if not donnees:
            continue

        chemin = os.path.join(DOSSIER_SORTIE, f"comparaison_{nom_axe}.png")
        figures.append(figure_axe_categoriel(config, donnees, chemin))
        synthese[nom_axe] = synthese_categorielle(nom_axe, donnees)

    chemin_json = os.path.join(DOSSIER_SORTIE, "comparaison.json")
    with open(chemin_json, "w", encoding="utf-8") as fh:
        json.dump(synthese, fh, indent=2, ensure_ascii=False)

    print(f"\n== Synthèse -> {chemin_json}")
    for f in figures:
        print(f"== Figure   -> {f}")


if __name__ == "__main__":
    main()
