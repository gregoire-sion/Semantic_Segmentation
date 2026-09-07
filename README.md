"""
MOTEUR DE BALAYAGE — commun à tous les axes

Idée centrale : un axe de généralisation (bruit, offset initial, famille
de commande...) ne diffère d'un autre QUE par la façon de fabriquer une
trajectoire à un réglage donné. Tout le reste — lancer les filtres,
calculer le Delta_dB, tracer, repérer les seuils — est identique.

Ce fichier écrit ce "tout le reste" une seule fois. Chaque script d'axe
(voir balayage_bruit.py) fournit seulement :
  - la liste des valeurs à balayer
  - une fonction qui génère des trajectoires pour une valeur donnée
  - une fonction qui donne l'estimation de l'EKF de référence

On ne lit donc ce fichier qu'une fois ; ensuite on ne lit plus que des
scripts d'axe d'une centaine de lignes.
"""

import os
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from KalmanNet_Drones import run_knet
from metriques import GROUPES, mse_par_groupe, delta_db, moyenne_ic95


# ==========================================================================
# 1. LE BALAYAGE PROPREMENT DIT
# ==========================================================================

def balayer(sm, modeles, valeurs, generer_trajectoires, estimer_ekf):
    """Parcourt les valeurs de l'axe et mesure les deux filtres.

    Paramètres
    ----------
    sm                   : le SystemModel
    modeles              : dict {seed: modèle KalmanNet}
    valeurs              : liste des réglages à tester
    generer_trajectoires : fonction(valeur) -> liste de (X, Y, U, M)
    estimer_ekf          : fonction(valeur, Y, U, M) -> estimation EKF

    Renvoie une liste de "points", un par valeur balayée.
    """
    points = []

    for valeur in valeurs:
        debut = time.time()
        trajectoires = generer_trajectoires(valeur)

        # L'EKF ne dépend d'aucun modèle appris : on le calcule une seule
        # fois par valeur, et non une fois par graine. Cela divise le
        # temps de calcul par le nombre de modèles.
        mse_ekf = [mse_par_groupe(estimer_ekf(valeur, Y, U, M), X)
                   for X, Y, U, M in trajectoires]

        point = {"valeur": valeur,
                 "n_trajectoires": len(trajectoires),
                 "par_seed": {}}

        for seed, model in modeles.items():
            # Les mêmes trajectoires servent à tous les modèles :
            # comparaison appariée, donc moins de bruit sur les écarts.
            mse_knet = [mse_par_groupe(run_knet(sm, model, Y, U, M), X)
                        for X, Y, U, M in trajectoires]

            deltas = [delta_db(k["position"], e["position"])
                      for k, e in zip(mse_knet, mse_ekf)]
            moyenne, ic95 = moyenne_ic95(deltas)

            resultat_seed = {"delta_db": moyenne, "ic95": ic95}
            for nom in GROUPES:
                resultat_seed[f"mse_knet_{nom}"] = float(
                    np.mean([m[nom] for m in mse_knet]))
            point["par_seed"][str(seed)] = resultat_seed

        for nom in GROUPES:
            point[f"mse_ekf_{nom}"] = float(np.mean([m[nom] for m in mse_ekf]))

        points.append(point)

        moyenne_axes = np.mean([point["par_seed"][str(s)]["delta_db"]
                                for s in modeles])
        print(f"   valeur={valeur:>6} | Delta_dB moyen = "
              f"{moyenne_axes:+6.2f} dB | {time.time() - debut:.0f} s")

    return points


# ==========================================================================
# 2. REPÈRES DE SEUIL
# ==========================================================================
# Rappel : l'objet d'étude est la FORME de la courbe, pas ces repères.
# Ils servent uniquement à produire un chiffre citable pour comparer deux
# modèles à l'étape finale.

def franchissement(valeurs, deltas, seuil):
    """Valeur de l'axe où la courbe franchit le seuil.

    Interpolation linéaire entre les deux points qui encadrent le
    changement de signe de (delta - seuil).

    Renvoie None si la courbe ne franchit jamais le seuil dans la plage
    balayée : on préfère afficher "hors plage" plutôt qu'extrapoler
    au-delà de ce qui a été mesuré.
    """
    ecarts = [d - seuil for d in deltas]
    for i in range(len(valeurs) - 1):
        e0, e1 = ecarts[i], ecarts[i + 1]
        if e0 == 0:
            return float(valeurs[i])
        if e0 * e1 < 0:
            x0, x1 = valeurs[i], valeurs[i + 1]
            return float(x0 + (x1 - x0) * (0 - e0) / (e1 - e0))
    return None


def resumer(points, seeds, seuils):
    """Franchissements par graine et sur la courbe moyenne."""
    valeurs = [p["valeur"] for p in points]
    courbe_moyenne = [
        float(np.mean([p["par_seed"][str(s)]["delta_db"] for s in seeds]))
        for p in points]

    resume = {"courbe_moyenne": courbe_moyenne, "seuils": {}}
    for seuil in seuils:
        par_seed = {}
        for s in seeds:
            deltas = [p["par_seed"][str(s)]["delta_db"] for p in points]
            par_seed[str(s)] = franchissement(valeurs, deltas, seuil)
        resume["seuils"][str(seuil)] = {
            "par_seed": par_seed,
            "sur_courbe_moyenne": franchissement(valeurs, courbe_moyenne, seuil),
        }

    print("\n== Repères de franchissement ==")
    for seuil in seuils:
        bloc = resume["seuils"][str(seuil)]
        f = bloc["sur_courbe_moyenne"]
        texte = f"{f:+.2f}" if f is not None else "hors plage balayée"
        etiquette = "parité (0 dB)" if seuil == 0 else f"seuil +{seuil:g} dB"
        print(f"   {etiquette:<16} | courbe moyenne : {texte}")
        for s, v in bloc["par_seed"].items():
            t = f"{v:+.2f}" if v is not None else "hors plage"
            print(f"      graine {s:<6} : {t}")

    return resume


# ==========================================================================
# 3. FIGURE
# ==========================================================================

def tracer(points, seeds, titre, label_x, valeur_entrainement,
           chemin_figure, seuil_repere=3.0):
    """Deux panneaux : performance relative et performance absolue.

    Le second panneau n'est pas décoratif. Delta_dB est un RAPPORT, donc
    sa référence bouge : l'EKF se dégrade lui aussi hors distribution.
    Une courbe Delta_dB plate peut donc vouloir dire "les deux filtres
    s'effondrent ensemble" et non "KalmanNet tient bon". Seule la MSE
    absolue permet de trancher.
    """
    valeurs = [p["valeur"] for p in points]
    fig, (ax_rel, ax_abs) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panneau gauche : Delta_dB, avec bande d'incertitude ---
    for seed in seeds:
        d = np.array([p["par_seed"][str(seed)]["delta_db"] for p in points])
        ic = np.array([p["par_seed"][str(seed)]["ic95"] for p in points])
        ax_rel.plot(valeurs, d, marker="o", lw=1.8, label=f"graine {seed}")
        ax_rel.fill_between(valeurs, d - ic, d + ic, alpha=0.15)

    ax_rel.axhline(0, color="k", lw=1.2, label="parité avec l'EKF")
    ax_rel.axhline(seuil_repere, color="crimson", ls="-.", lw=1.4,
                   label=f"repère +{seuil_repere:g} dB")
    if valeur_entrainement is not None:
        ax_rel.axvline(valeur_entrainement, color="gray", ls="--", lw=1,
                       label="point d'entraînement")
    ax_rel.set_xlabel(label_x)
    ax_rel.set_ylabel(r"$\Delta_{dB}$ position   (< 0 : KNet meilleur)")
    ax_rel.set_title("Performance relative")
    ax_rel.grid(True, ls=":", alpha=0.7)
    ax_rel.legend(fontsize=8)

    # --- Panneau droit : MSE absolue, échelle logarithmique ---
    for seed in seeds:
        mse = [p["par_seed"][str(seed)]["mse_knet_position"] for p in points]
        ax_abs.plot(valeurs, mse, marker="o", lw=1.8, label=f"KNet graine {seed}")
    mse_ekf = [p["mse_ekf_position"] for p in points]
    ax_abs.plot(valeurs, mse_ekf, marker="s", lw=2.2, color="k", ls="--",
                label="EKF de référence")
    if valeur_entrainement is not None:
        ax_abs.axvline(valeur_entrainement, color="gray", ls="--", lw=1)
    ax_abs.set_yscale("log")
    ax_abs.set_xlabel(label_x)
    ax_abs.set_ylabel("MSE position (échelle log)")
    ax_abs.set_title("Performance absolue")
    ax_abs.grid(True, ls=":", alpha=0.7, which="both")
    ax_abs.legend(fontsize=8)

    fig.suptitle(titre)
    fig.tight_layout()
    fig.savefig(chemin_figure, dpi=140)
    plt.close(fig)
    return chemin_figure


# ==========================================================================
# 4. ORCHESTRATION D'UN AXE COMPLET
# ==========================================================================

def lancer_un_axe(sm, modeles, config):
    """Enchaîne balayage -> repères -> figure -> sauvegarde JSON.

    config est un dictionnaire fourni par le script d'axe. Voir
    balayage_bruit.py pour un exemple commenté de son contenu.
    """
    os.makedirs(config["dossier_sortie"], exist_ok=True)
    seeds = sorted(modeles)
    seuils = config.get("seuils", (0.0, 3.0))
    nom = config["nom"]

    debut = time.time()
    points = balayer(sm, modeles, config["valeurs"],
                     config["generer_trajectoires"], config["estimer_ekf"])
    resume = resumer(points, seeds, seuils)

    figure = tracer(points, seeds, config["titre"], config["label_x"],
                    config.get("valeur_entrainement"),
                    os.path.join(config["dossier_sortie"], nom + ".png"),
                    seuil_repere=config.get("seuil_repere", 3.0))

    sortie = {
        "axe": nom,
        "jeu": config.get("jeu", "dev"),
        "seeds": seeds,
        "valeurs": config["valeurs"],
        "seuils": list(seuils),
        "points": points,
        "resume": resume,
        "duree_s": round(time.time() - debut, 1),
    }
    chemin_json = os.path.join(config["dossier_sortie"], nom + ".json")
    with open(chemin_json, "w", encoding="utf-8") as fh:
        json.dump(sortie, fh, indent=2, ensure_ascii=False)

    print(f"\n== Résultats -> {chemin_json}")
    print(f"== Figure    -> {figure}")
    return sortie
