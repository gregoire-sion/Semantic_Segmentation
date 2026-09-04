"""
LE MOTEUR DE BALAYAGE - commun à tous les axes de l'étude.

Idée centrale : un axe de généralisation (bruit, offset initial, famille
de commande...) ne diffère d'un autre QUE par la façon de fabriquer une
trajectoire à un réglage donné. Tout le reste — lancer les filtres,
calculer le Delta_dB, chercher la frontière, tracer — est identique.

Ce fichier écrit ce "tout le reste" une seule fois. Chaque axe (voir
axe_bruit.py) se contente de fournir :
  - une liste de valeurs à balayer
  - une fonction qui, pour une valeur, renvoie des trajectoires
  - une fonction qui, pour une valeur et une trajectoire, renvoie l'EKF
    de référence (nominal partout, sauf sur l'axe bruit où l'on veut
    l'oracle)

On ne lit donc ce fichier qu'une fois ; ensuite on ne lit plus que des
axes de 30 lignes.
"""

import os
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from KalmanNet_Drones import run_knet
from commun import GROUPES, mse_par_groupe, delta_db, moyenne_ic95


# ==========================================================================
# 1. BALAYAGE
# ==========================================================================

def balayer(sm, modeles, valeurs, generer_lot, ekf_reference):
    """Parcourt les valeurs de l'axe et mesure les deux filtres.

    Paramètres
    ----------
    sm            : le SystemModel
    modeles       : dict {seed: KalmanNet}
    valeurs       : liste des réglages à tester (ex. niveaux de bruit)
    generer_lot   : fonction(valeur) -> liste de (X, Y, U, M)
    ekf_reference : fonction(valeur, Y, U, M) -> estimation EKF de référence

    Renvoie une liste de points, un par valeur balayée.
    """
    points = []
    for valeur in valeurs:
        t0 = time.time()
        lot = generer_lot(valeur)

        # EKF de référence : ne dépend pas du modèle, calculé une seule fois.
        mse_ekf = [mse_par_groupe(ekf_reference(valeur, Y, U, M), X)
                   for X, Y, U, M in lot]

        point = {"valeur": valeur, "n_eval": len(lot), "par_seed": {}}

        for seed, model in modeles.items():
            mse_knet = [mse_par_groupe(run_knet(sm, model, Y, U, M), X)
                        for X, Y, U, M in lot]

            deltas = [delta_db(k["position"], e["position"])
                      for k, e in zip(mse_knet, mse_ekf)]
            moy, ic = moyenne_ic95(deltas)

            entree = {"delta_db": moy, "ic95": ic}
            for g in GROUPES:
                entree[f"mse_knet_{g}"] = float(np.mean([m[g] for m in mse_knet]))
            point["par_seed"][str(seed)] = entree

        for g in GROUPES:
            point[f"mse_ekf_{g}"] = float(np.mean([m[g] for m in mse_ekf]))
        points.append(point)

        moy_axes = np.mean([point["par_seed"][str(s)]["delta_db"]
                            for s in modeles])
        print(f"   valeur={valeur:>6} | Delta_dB moyen = "
              f"{moy_axes:+6.2f} dB | {time.time()-t0:.0f} s")

    return points


# ==========================================================================
# 2. FRONTIÈRE
# ==========================================================================

def frontiere(valeurs, deltas, seuil=0.0):
    """Valeur de l'axe où Delta_dB franchit le seuil (interpolation linéaire).

    Le seuil est le niveau de dégradation jugé inacceptable. On ne cherche
    pas le point où KalmanNet perd son avantage (seuil 0), mais celui où il
    cesse d'être utilisable : seuil = +3 dB signifie "l'erreur de KalmanNet
    vaut le double de celle de l'EKF".

    Renvoie None si la courbe ne franchit jamais le seuil dans la plage
    balayée : on préfère dire "hors plage" plutôt qu'extrapoler.
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


def resumer_frontieres(points, seeds, seuils=(0.0, 3.0)):
    """Frontières par graine et sur la courbe moyenne, pour chaque seuil."""
    valeurs = [p["valeur"] for p in points]
    courbe_moy = [float(np.mean([p["par_seed"][str(s)]["delta_db"] for s in seeds]))
                  for p in points]

    resume = {"courbe_moyenne": courbe_moy, "seuils": {}}
    for seuil in seuils:
        par_seed = {}
        for s in seeds:
            d = [p["par_seed"][str(s)]["delta_db"] for p in points]
            par_seed[str(s)] = frontiere(valeurs, d, seuil)
        resume["seuils"][str(seuil)] = {
            "par_seed": par_seed,
            "sur_moyenne": frontiere(valeurs, courbe_moy, seuil),
        }

    print("\n== Frontières ==")
    for seuil in seuils:
        bloc = resume["seuils"][str(seuil)]
        f = bloc["sur_moyenne"]
        txt = f"{f:+.2f}" if f is not None else "hors plage"
        etiquette = "parite" if seuil == 0 else f"seuil +{seuil:g} dB"
        print(f"   {etiquette:<14} | courbe moyenne : {txt}")
        for s, v in bloc["par_seed"].items():
            t = f"{v:+.2f}" if v is not None else "hors plage"
            print(f"      graine {s:<6} : {t}")
    return resume


# ==========================================================================
# 3. FIGURES
# ==========================================================================

def tracer(points, seeds, titre, label_x, valeur_ref, chemin,
           seuil_principal=3.0):
    """Deux panneaux : Delta_dB (relatif) et MSE absolue.

    Le second panneau n'est pas décoratif. Delta_dB est un rapport, donc sa
    référence bouge : l'EKF se dégrade lui aussi hors distribution. Une
    courbe Delta_dB plate peut donc signifier "les deux se dégradent
    ensemble" et non "KalmanNet tient bon". La MSE absolue lève l'ambiguïté.
    """
    valeurs = [p["valeur"] for p in points]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panneau gauche : Delta_dB ---
    for s in seeds:
        d = np.array([p["par_seed"][str(s)]["delta_db"] for p in points])
        ic = np.array([p["par_seed"][str(s)]["ic95"] for p in points])
        ax.plot(valeurs, d, marker="o", lw=1.8, label=f"graine {s}")
        ax.fill_between(valeurs, d - ic, d + ic, alpha=0.15)

    ax.axhline(0, color="k", lw=1.2, label="parité avec l'EKF")
    ax.axhline(seuil_principal, color="crimson", ls="-.", lw=1.4,
               label=f"seuil +{seuil_principal:g} dB")
    if valeur_ref is not None:
        ax.axvline(valeur_ref, color="gray", ls="--", lw=1,
                   label="point d'entraînement")
    ax.set_xlabel(label_x)
    ax.set_ylabel(r"$\Delta_{dB}$ position   (< 0 : KNet meilleur)")
    ax.set_title("Performance relative")
    ax.grid(True, ls=":", alpha=0.7)
    ax.legend(fontsize=8)

    # --- Panneau droit : MSE absolue ---
    for s in seeds:
        mk = [p["par_seed"][str(s)]["mse_knet_position"] for p in points]
        ax2.plot(valeurs, mk, marker="o", lw=1.8, label=f"KNet graine {s}")
    me = [p["mse_ekf_position"] for p in points]
    ax2.plot(valeurs, me, marker="s", lw=2.2, color="k", ls="--",
             label="EKF de référence")
    ax2.set_yscale("log")
    ax2.set_xlabel(label_x)
    ax2.set_ylabel("MSE position (échelle log)")
    ax2.set_title("Performance absolue")
    ax2.grid(True, ls=":", alpha=0.7, which="both")
    ax2.legend(fontsize=8)

    fig.suptitle(titre)
    fig.tight_layout()
    fig.savefig(chemin, dpi=140)
    plt.close(fig)
    return chemin


# ==========================================================================
# 4. ORCHESTRATION D'UN AXE COMPLET
# ==========================================================================

def lancer_axe(sm, modeles, config):
    """Enchaîne balayage -> frontière -> figure -> sauvegarde JSON.

    config est un dictionnaire fourni par le fichier d'axe. Voir
    axe_bruit.py pour un exemple commenté de son contenu.
    """
    os.makedirs(config["out_dir"], exist_ok=True)
    seeds = sorted(modeles)
    print(f">> {len(seeds)} modèle(s) : {seeds}\n")

    t0 = time.time()
    points = balayer(sm, modeles, config["valeurs"],
                     config["generer_lot"], config["ekf_reference"])

    seuils = config.get("seuils", (0.0, 3.0))
    resume = resumer_frontieres(points, seeds, seuils)

    figure = tracer(points, seeds, config["titre"], config["label_x"],
                    config.get("valeur_ref"),
                    os.path.join(config["out_dir"], config["nom"] + ".png"),
                    seuil_principal=config.get("seuil_principal", 3.0))

    sortie = {
        "axe": config["nom"],
        "seeds": seeds,
        "valeurs": config["valeurs"],
        "points": points,
        "seuils": list(seuils),
        "resume_frontieres": resume,
        "duree_s": round(time.time() - t0, 1),
    }
    chemin_json = os.path.join(config["out_dir"], config["nom"] + ".json")
    with open(chemin_json, "w", encoding="utf-8") as fh:
        json.dump(sortie, fh, indent=2, ensure_ascii=False)

    print(f"\n== Résultats -> {chemin_json}")
    print(f"== Figure    -> {figure}")
    return sortie

