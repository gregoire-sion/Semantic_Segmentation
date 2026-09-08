"""
ÉTAPE 3 — ATTRIBUTION : POURQUOI KALMANNET DIVERGE À FORT OFFSET INITIAL

Le balayage de l'axe offset montre QUE le réseau diverge au-delà d'une
amplitude d'environ 1,0. Ce script cherche POURQUOI, en regardant à
l'intérieur du réseau pendant qu'il filtre.


L'HYPOTHÈSE TESTÉE
------------------
Dans _features() du réseau, les quatre features sont normalisées :

    norm = lambda v: F.normalize(v, p=2, dim=1, eps=1e-12)

Leur norme vaut donc TOUJOURS 1, quelle que soit l'ampleur de l'erreur
réelle. Le réseau ne perçoit que la DIRECTION de son erreur, jamais son
intensité.

Mais la mise à jour, elle, utilise l'innovation BRUTE :

    dy   = (y - m1y) * mask        <- pas normalisée
    inov = KG @ dy

Si l'offset initial est grand, dy est grand. Le réseau calcule pourtant
son gain à partir d'une feature d'apparence ordinaire, et ce gain est
ensuite appliqué à une innovation énorme. Rien ne borne le résultat.

L'EKF ne peut pas défaillir ainsi : son gain vient de K = P H' S^-1, donc
il SAIT que son incertitude initiale est grande et corrige de façon
proportionnée.


CE QUE LE SCRIPT PRODUIT
------------------------
Figure 1 (dynamique.png)  : ce qui se passe pas par pas
Figure 2 (features.png)   : les features sortent-elles du domaine appris,
                            en norme ou seulement en direction ?
Un résumé chiffré en console et un fichier JSON.

Le test décisif est sur la figure 2 : si la norme des features reste
plate à 1 et que seul le cosinus décroche, l'hypothèse est confirmée.


LIMITE
------
Une seule graine est instrumentée. C'est assez pour identifier un
mécanisme, pas pour affirmer qu'il est général. Si les figures
confirment l'hypothèse, refaire tourner avec SEED_MODELE = 1234 puis 7.

À LANCER :
    python attribution_offset.py
"""

import os
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from KalmanNet_Drones import CFG, SystemModel, generate_trajectory
from metriques import GROUPES
from chargement_modeles import chemin_checkpoint, charger_un_modele


# ==========================================================================
# 1. RÉGLAGES
# ==========================================================================

DOSSIER_RUNS = "./runs"
ARCHI = "archi2"
SEED_MODELE = 42                 # une seule graine suffit pour un diagnostic

# Trois régimes : en distribution, à la frontière, très au-delà.
ECHELLES = [0.3, 1.0, 2.0]
ECHELLE_REFERENCE = 0.3          # le point d'entraînement

N_TRAJECTOIRES = 20
SEED_DIAG = 24250

DOSSIER_SORTIE = "./runs/attribution_offset"

IDX_POSITION = GROUPES["position"]

COULEURS = {0.3: "tab:green", 1.0: "tab:orange", 2.0: "tab:red"}

# Les grandeurs relevées à chaque pas de temps.
GRANDEURS = ("norme_dy", "norme_KG", "norme_inov",
             "depassement", "mse_position")


# ==========================================================================
# 2. INFÉRENCE INSTRUMENTÉE
# ==========================================================================

def filtrer_en_observant(sm, model, Y, U, M, X_vrai):
    """Filtre une trajectoire et relève l'intérieur du réseau à chaque pas.

    La boucle reproduit exactement celle de run_knet : même ordre d'appel,
    même mise en forme des tenseurs. Seuls des relevés sont ajoutés.

    Note de lecture : on appelle step_prior(u) AVANT model(...), pour
    pouvoir lire les features telles qu'elles seront calculées. model()
    rappellera step_prior en interne — c'est déterministe, donc sans
    conséquence, mais il faut le savoir pour ne pas s'en étonner.
    """
    T = U.shape[0]
    model.eval()
    model.init_sequence(sm.x0, 1)

    releves = {nom: [] for nom in GRANDEURS}
    releves["features"] = []

    x_estime = torch.zeros(T + 1, sm.m, 1, device=sm.device)
    x_estime[0] = sm.x0

    with torch.no_grad():
        for k in range(1, T + 1):
            y = Y[k].unsqueeze(0)
            u = U[k - 1].unsqueeze(0)

            # (a) Ce que le réseau voit en entrée, avant la mise à jour
            model.step_prior(u)
            f1, f2, f3, f4 = model._features(y)
            features = torch.cat([f1, f2, f3, f4], dim=1).squeeze(0)

            # (b) L'innovation brute, celle réellement utilisée
            dy = (y - model.m1y) * M[k].reshape(1, sm.n, 1)

            # (c) L'erreur de la prédiction, avant correction
            erreur_avant = torch.linalg.norm(model.m1x_prior - X_vrai[k]).item()

            # (d) Le pas complet
            x_estime[k] = model(y, u, M[k]).squeeze(0)

            # (e) La correction qui vient d'être appliquée
            KG = model.KGain
            correction = torch.bmm(KG, dy)

            erreur_position = ((x_estime[k][IDX_POSITION, 0]
                                - X_vrai[k][IDX_POSITION, 0]) ** 2).mean()

            releves["norme_dy"].append(torch.linalg.norm(dy).item())
            releves["norme_KG"].append(torch.linalg.norm(KG).item())
            releves["norme_inov"].append(torch.linalg.norm(correction).item())
            # Dépassement : la correction est-elle du même ordre de grandeur
            # que l'erreur qu'elle est censée corriger ? Un rapport proche
            # de 1 est sain ; très supérieur à 1 signale un sur-ajustement.
            releves["depassement"].append(
                torch.linalg.norm(correction).item() / max(erreur_avant, 1e-9))
            releves["mse_position"].append(erreur_position.item())
            releves["features"].append(features.cpu().numpy())

    releves["features"] = np.stack(releves["features"])
    return releves


def collecter(sm, model, echelle, n):
    """Lance l'inférence instrumentée sur n trajectoires à une échelle donnée.

    generate_trajectory lit CFG.INIT_OFFSET_SCALE au moment de l'appel :
    il suffit de régler ce paramètre avant de générer.
    """
    CFG.INIT_OFFSET_SCALE = echelle
    CFG.INIT_OFFSET_P0 = (echelle > 0.0)
    rng = np.random.default_rng(SEED_DIAG + int(round(echelle * 100)))

    trajectoires = []
    for _ in range(n):
        X, Y, U, M = generate_trajectory(sm, rng, r_scale=1.0)
        trajectoires.append(filtrer_en_observant(sm, model, Y, U, M, X))
    return trajectoires


# ==========================================================================
# 3. AGRÉGATION
# ==========================================================================

def mediane_pas_par_pas(trajectoires, nom_grandeur):
    """Médiane, sur les trajectoires, d'une grandeur relevée à chaque pas.

    On prend la MÉDIANE et non la moyenne : une seule trajectoire
    divergente à 1e15 écraserait toute moyenne et rendrait les courbes
    plates et illisibles. La médiane donne le comportement typique.
    En contrepartie elle masque les cas extrêmes — c'est un compromis
    assumé, à mentionner si on présente ces courbes.
    """
    empile = np.stack([t[nom_grandeur] for t in trajectoires])
    return np.median(empile, axis=0)


def features_moyennes(trajectoires):
    """Vecteur de features moyen à chaque pas de temps.

    Forme du résultat : (nombre de pas, dimension des features).
    """
    return np.mean(np.stack([t["features"] for t in trajectoires]), axis=0)


def normaliser_lignes(matrice):
    """Ramène chaque ligne à une norme de 1.

    Étape séparée pour que le calcul du cosinus, plus bas, se lise en
    une seule ligne évidente.
    """
    normes = np.linalg.norm(matrice, axis=1)
    return matrice / (normes[:, None] + 1e-12), normes


def cosinus_avec_reference(features, features_reference):
    """Similarité cosinus, pas par pas, entre deux séries de features.

    Vaut 1 si les deux vecteurs pointent dans la même direction, 0 s'ils
    sont perpendiculaires. Comme les features sont déjà normalisées par
    le réseau, la direction est la SEULE chose qui peut varier — d'où
    l'intérêt de cette mesure.
    """
    f_unitaire, _ = normaliser_lignes(features)
    ref_unitaire, _ = normaliser_lignes(features_reference)
    return np.sum(f_unitaire * ref_unitaire, axis=1)


# ==========================================================================
# 4. FIGURES
# ==========================================================================

def figure_dynamique(dynamiques, chemin):
    """Quatre panneaux : ce qui se passe pas par pas, selon l'offset."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    panneaux = [
        ("norme_dy", "Innovation brute  ||dy||", axes[0][0]),
        ("norme_KG", "Gain appris  ||KG||", axes[0][1]),
        ("depassement", "Dépassement  ||correction|| / ||erreur||", axes[1][0]),
        ("mse_position", "MSE position instantanée (30 premiers pas)",
         axes[1][1]),
    ]

    for nom, titre, ax in panneaux:
        for echelle in ECHELLES:
            ax.plot(dynamiques[echelle][nom], lw=1.6,
                    color=COULEURS.get(echelle), label=f"offset {echelle}")
        ax.set_yscale("log")
        ax.set_xlabel("pas de temps")
        ax.set_title(titre, fontsize=11)
        ax.grid(True, ls=":", alpha=0.7, which="both")
        ax.legend(fontsize=8)

    # Zoom : si l'hypothèse du transitoire initial est bonne, tout se joue
    # dans les tout premiers pas.
    axes[1][1].set_xlim(0, 30)

    fig.suptitle("Attribution — dynamique interne selon l'offset initial")
    fig.tight_layout()
    fig.savefig(chemin, dpi=140)
    plt.close(fig)
    return chemin


def figure_features(analyses, chemin):
    """Le test décisif : norme constante, direction qui décroche ?"""
    fig, (ax_norme, ax_cos) = plt.subplots(1, 2, figsize=(13, 5))

    for echelle in ECHELLES:
        ax_norme.plot(analyses[echelle]["normes"], lw=1.6,
                      color=COULEURS.get(echelle), label=f"offset {echelle}")
    ax_norme.set_xlabel("pas de temps")
    ax_norme.set_ylabel("norme du vecteur de features")
    ax_norme.set_title("Norme des features\n"
                       "attendue constante : elles sont normalisées L2",
                       fontsize=10)
    ax_norme.grid(True, ls=":", alpha=0.7)
    ax_norme.legend(fontsize=8)

    for echelle in ECHELLES:
        ax_cos.plot(analyses[echelle]["cosinus"], lw=1.6,
                    color=COULEURS.get(echelle), label=f"offset {echelle}")
    ax_cos.axhline(1.0, color="k", lw=1, ls="--")
    ax_cos.set_xlabel("pas de temps")
    ax_cos.set_ylabel("similarité cosinus avec la référence")
    ax_cos.set_title("Direction des features\n"
                     "c'est elle qui peut sortir du domaine appris",
                     fontsize=10)
    ax_cos.grid(True, ls=":", alpha=0.7)
    ax_cos.legend(fontsize=8)

    fig.suptitle("Attribution — les features vues par le réseau")
    fig.tight_layout()
    fig.savefig(chemin, dpi=140)
    plt.close(fig)
    return chemin


# ==========================================================================
# 5. AFFICHAGE DU RÉSUMÉ
# ==========================================================================

def afficher_resume(resume):
    print("\n== Dynamique (médiane sur les 20 premiers pas) ==")
    print(f"{'offset':>8} {'||dy||':>12} {'||KG||':>12} "
          f"{'depassement':>13} {'MSE pos':>12}")
    for echelle in ECHELLES:
        l = resume[str(echelle)]
        print(f"{echelle:>8} {l['norme_dy']:>12.3f} {l['norme_KG']:>12.3f} "
              f"{l['depassement']:>13.3f} {l['mse_position']:>12.4f}")

    print("\n== Features (moyenne sur toute la trajectoire) ==")
    print(f"{'offset':>8} {'norme':>12} {'cosinus vs ref':>16}")
    for echelle in ECHELLES:
        l = resume[str(echelle)]
        print(f"{echelle:>8} {l['norme_features']:>12.4f} "
              f"{l['cosinus_moyen']:>16.4f}")

    print("\nLecture : si la norme reste plate à 1 et que seul le cosinus")
    print("décroche, le réseau est bien aveugle à l'ampleur de son erreur.")


# ==========================================================================
# 6. PROGRAMME PRINCIPAL
# ==========================================================================

def main():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    torch.manual_seed(SEED_DIAG)
    np.random.seed(SEED_DIAG)

    CFG.TRAIN_CMD_RANDOMIZE = False

    print("== Attribution : axe offset initial ==")
    print(f"   échelles     : {ECHELLES}")
    print(f"   trajectoires : {N_TRAJECTOIRES} par échelle")
    print(f"   modèle       : baseline étroite, graine {SEED_MODELE}\n")

    sm = SystemModel()
    chemin = chemin_checkpoint(DOSSIER_RUNS, ARCHI, SEED_MODELE)
    if not os.path.exists(chemin):
        raise SystemExit(f"Checkpoint introuvable : {chemin}")
    model = charger_un_modele(sm, chemin, ARCHI)

    # --- Collecte ---------------------------------------------------------
    brut = {}
    for echelle in ECHELLES:
        print(f"   collecte offset {echelle} ...")
        brut[echelle] = collecter(sm, model, echelle, N_TRAJECTOIRES)

    # --- Dynamiques -------------------------------------------------------
    dynamiques = {
        echelle: {nom: mediane_pas_par_pas(brut[echelle], nom)
                  for nom in GRANDEURS}
        for echelle in ECHELLES
    }

    # --- Features ---------------------------------------------------------
    # La direction de référence est celle observée en distribution.
    features_ref = features_moyennes(brut[ECHELLE_REFERENCE])

    analyses = {}
    for echelle in ECHELLES:
        features = features_moyennes(brut[echelle])
        _, normes = normaliser_lignes(features)
        analyses[echelle] = {
            "normes": normes,
            "cosinus": cosinus_avec_reference(features, features_ref),
        }

    # --- Sorties ----------------------------------------------------------
    fig1 = figure_dynamique(dynamiques,
                            os.path.join(DOSSIER_SORTIE, "dynamique.png"))
    fig2 = figure_features(analyses,
                           os.path.join(DOSSIER_SORTIE, "features.png"))

    resume = {}
    for echelle in ECHELLES:
        d = dynamiques[echelle]
        resume[str(echelle)] = {
            "norme_dy": float(np.median(d["norme_dy"][:20])),
            "norme_KG": float(np.median(d["norme_KG"][:20])),
            "depassement": float(np.median(d["depassement"][:20])),
            "mse_position": float(np.median(d["mse_position"][:20])),
            "norme_features": float(np.mean(analyses[echelle]["normes"])),
            "cosinus_moyen": float(np.mean(analyses[echelle]["cosinus"])),
        }

    afficher_resume(resume)

    chemin_json = os.path.join(DOSSIER_SORTIE, "attribution_offset.json")
    with open(chemin_json, "w", encoding="utf-8") as fh:
        json.dump({"seed_modele": SEED_MODELE, "echelles": ECHELLES,
                   "resume": resume}, fh, indent=2, ensure_ascii=False)

    print(f"\n== Résumé   -> {chemin_json}")
    print(f"== Figure 1 -> {fig1}")
    print(f"== Figure 2 -> {fig2}")


if __name__ == "__main__":
    main()
