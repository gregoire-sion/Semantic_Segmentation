"""
Etude de generalisation de KalmanNet : jusqu'ou le reseau tient-il quand les
conditions de test s'eloignent des conditions d'entrainement ?

Ce fichier se lit de haut en bas, en cinq parties :

    1. Reglages          les constantes de l'etude
    2. Scenarios         la liste de toutes les conditions testees
    3. Evaluation        la boucle de calcul et la metrique
    4. Figures           les graphiques et le tableau de resultats
    5. main              l'enchainement complet

Metrique employee partout :

    Delta_dB = 10 log10( MSE_KNet / MSE_EKF )   negatif -> KalmanNet bat l'EKF
    div_rate = part des runs ou MSE_KNet depasse 100 fois celle de l'EKF

La comparaison est APPARIEE : pour un scenario donne, les memes trajectoires
servent a l'EKF et a tous les modeles. On raisonne en ratio et jamais en MSE
absolue, parce qu'un scenario plus difficile degrade aussi l'EKF : une MSE
absolue ferait croire a un merite qui vient en realite de la difficulte du cas.

Usage :
    python etude_generalisation.py            # evaluation complete
    python etude_generalisation.py --fumee    # version rapide, pour tester le code
"""

import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from KalmanNet_Drones import (CFG, SystemModel, EKF, KalmanNetNN,
                              generate_trajectory, corrupt_observations,
                              run_knet, rebind, CAPTEURS_GPS)
from ood_commands import build_command, FAMILY_REF, FAMILIES_TEST, LABELS_FR


# =========================================================================
# 1. REGLAGES
# =========================================================================

MODELES = {                          # nom affiche -> dossier des checkpoints
    "A (etroit)":    "./Dataset",
    "B (randomise)": "./Dataset_ood",
}
ARCHIS = ["archi1", "archi2"]

N_MC = 20                            # trajectoires tirees par scenario
SEED_EVAL = 12345
OUT_DIR = "./eval_generalisation"

FACTEUR_DIVERGENCE = 100.0           # run divergent si MSE_KNet > 100 x MSE_EKF

T_REF = 160                          # horizon d'entrainement (CFG.T)
DT = 0.1                             # pas de temps du modele (SystemModel.dt)
TRANSITOIRE_S = 2.0                  # duree des fenetres transitoire et etabli

# Drone 2 = le drone critique. Le filtre ne connait ni sa commande
# (B_filter, estim_d2=False) ni son modele d'acceleration (F_filter), et il
# n'a pas de GPS. C'est la que KalmanNet peut apporter quelque chose.
IDX_POS_D2 = slice(8, 10)
IDX_POS_ALL = [0, 1, 8, 9, 16, 17]

# Fenetres temporelles sur lesquelles la MSE est calculee. Noms fixes : on peut
# les retrouver par une simple recherche dans le fichier.
FENETRES = ["total", "transitoire", "etabli", "bloc_debut", "bloc_fin"]

GROUPES = ["commandes", "horizon", "capteurs", "bruit", "geometrie",
           "condition_initiale"]

COULEURS = {
    ("A (etroit)", "archi1"):    "#c44e52",
    ("A (etroit)", "archi2"):    "#8c2d30",
    ("B (randomise)", "archi1"): "#4c72b0",
    ("B (randomise)", "archi2"): "#2a4a7f",
}
STYLES = {"archi1": "--", "archi2": "-"}


# =========================================================================
# 2. SCENARIOS
# =========================================================================

def scenario(nom, axe, libelle, x=None,
             famille=FAMILY_REF, amplitude=None,
             T=T_REF, r_scale=1.0, q_scale=1.0, offset_scale=1.0,
             formation="triangle", ratio_gps=5,
             pannes=(), taux_aberrant=0.0,
             nominal=False, reference=False):
    """Decrit une condition de test, sous forme de dictionnaire.

    Les valeurs par defaut de cette signature SONT les conditions
    d'entrainement. Un scenario ne precise donc que ce qu'il fait varier, et
    la signature ci-dessus enumere tout ce qui peut varier dans l'etude.

      nom, axe, libelle : identification et affichage
      x                 : abscisse quand l'axe est un balayage (None sinon)
      famille, amplitude: commande, passees a build_command()
      T, r_scale, q_scale, offset_scale : passes a generate_trajectory()
      formation, ratio_gps              : passes a SystemModel()
      pannes, taux_aberrant             : passes a corrupt_observations()
      reference : point de reference DE CET AXE
      nominal   : strictement les conditions d'entrainement
    """
    return {"nom": nom, "axe": axe, "libelle": libelle, "x": x,
            "famille": famille, "amplitude": amplitude,
            "T": T, "r_scale": r_scale, "q_scale": q_scale,
            "offset_scale": offset_scale,
            "formation": formation, "ratio_gps": ratio_gps,
            "pannes": pannes, "taux_aberrant": taux_aberrant,
            "nominal": nominal, "reference": reference}


def axe(nom, groupe, titre, xlabel, scenarios):
    """Un axe de decalage. xlabel vide -> axe categoriel, trace en barres."""
    return {"nom": nom, "groupe": groupe, "titre": titre,
            "xlabel": xlabel, "scenarios": scenarios}


# --- Axe 1 : familles de commandes jamais vues -------------------------
scenarios_commandes = [
    scenario("commandes_ref", "commandes", LABELS_FR[FAMILY_REF],
             nominal=True, reference=True)]
for famille in FAMILIES_TEST:
    scenarios_commandes.append(
        scenario("commandes_" + famille, "commandes", LABELS_FR[famille],
                 famille=famille))

# --- Axe 2 : extrapolation en amplitude de commande --------------------
# Seul axe dont la reference n'est PAS la condition nominale : il porte sur la
# famille creneaux, jamais vue a l'entrainement. A = 1 y est le point de
# reference de l'axe, mais pas un point nominal.
scenarios_amplitude = []
for amplitude in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    scenarios_amplitude.append(
        scenario(f"amplitude_{amplitude:g}", "amplitude",
                 f"A = {amplitude:g}", x=amplitude,
                 famille="creneaux", amplitude=amplitude,
                 reference=(amplitude == 1.0)))

# --- Axe 3 : horizon plus long que celui vu ----------------------------
scenarios_horizon = [
    scenario("horizon_160", "horizon", "T = 160 (entrainement)", x=160,
             nominal=True, reference=True)]
for T in (320, 480, 960):
    scenarios_horizon.append(
        scenario(f"horizon_{T}", "horizon", f"T = {T} ({T // T_REF}x)",
                 x=T, T=T))

# --- Axe 4 : cadence du GPS et des distances ---------------------------
scenarios_cadence = [
    scenario("cadence_5", "cadence", "ratio_gps = 5 (entrainement)", x=5,
             nominal=True, reference=True)]
for ratio in (1, 2, 10, 20):
    scenarios_cadence.append(
        scenario(f"cadence_{ratio}", "cadence", f"ratio_gps = {ratio}",
                 x=ratio, ratio_gps=ratio))


# --- Axe 5 : panne totale du GPS et des distances ----------------------
def fenetre_de_panne(duree_s):
    """Une panne centree sur la trajectoire, exprimee en pas de temps.

    CAPTEURS_GPS couvre le GPS du drone 1 ET les 3 distances inter-drones :
    pendant la fenetre il ne reste donc que l'accelerometre.
    """
    n_pas = int(round(duree_s / DT))
    k_debut = (T_REF - n_pas) // 2
    return [(k_debut, k_debut + n_pas - 1, CAPTEURS_GPS)]


scenarios_panne = [
    scenario("panne_0s", "panne", "aucune panne", x=0.0,
             nominal=True, reference=True)]
for duree in (2.0, 5.0, 10.0):
    scenarios_panne.append(
        scenario(f"panne_{duree:g}s", "panne", f"panne de {duree:g} s",
                 x=duree, pannes=fenetre_de_panne(duree)))

# --- Axe 6 : mesures aberrantes ----------------------------------------
scenarios_aberrations = [
    scenario("aberrations_0", "aberrations", "aucune aberration", x=0.0,
             nominal=True, reference=True)]
for taux in (0.01, 0.02, 0.05):
    scenarios_aberrations.append(
        scenario(f"aberrations_{taux:g}", "aberrations",
                 f"{100 * taux:g} % aberrantes", x=taux, taux_aberrant=taux))

# --- Axe 7 : bruit de mesure -------------------------------------------
# Le modele B a ete entraine avec 1/r^2 tire dans [-10, 30] dB.
# -30 et -20 dB, comme +40 et +50 dB, sont donc hors plage.
scenarios_bruit_r = []
for db in (-30, -20, -10, 0, 10, 20, 30, 40, 50):
    est_nominal = (db == 0)
    scenarios_bruit_r.append(
        scenario(f"bruit_r_{db:+d}dB", "bruit_r", f"{db:+d} dB", x=db,
                 r_scale=10.0 ** (-db / 20.0),
                 nominal=est_nominal, reference=est_nominal))

# --- Axe 8 : bruit de process (jamais varie a l'entrainement) ----------
scenarios_bruit_q = []
for facteur in (0.2, 1.0, 5.0, 25.0):
    est_nominal = (facteur == 1.0)
    scenarios_bruit_q.append(
        scenario(f"bruit_q_{facteur:g}", "bruit_q", f"q x {facteur:g}",
                 x=facteur, q_scale=facteur,
                 nominal=est_nominal, reference=est_nominal))

# --- Axe 9 : geometrie de la formation ---------------------------------
LIBELLES_FORMATION = {
    "triangle": "triangle (entrainement)",
    "ligne":    "colineaire (d13 = d12 + d23)",
    "serree":   "formation serree",
    "large":    "formation large",
}
scenarios_geometrie = [
    scenario("geometrie_triangle", "geometrie",
             LIBELLES_FORMATION["triangle"], nominal=True, reference=True)]
for forme in ("ligne", "serree", "large"):
    scenarios_geometrie.append(
        scenario("geometrie_" + forme, "geometrie",
                 LIBELLES_FORMATION[forme], formation=forme))

# --- Axe 10 : erreur sur l'etat initial --------------------------------
scenarios_condition_initiale = []
for offset in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
    est_nominal = (offset == 1.0)
    scenarios_condition_initiale.append(
        scenario(f"condition_initiale_{offset:g}", "condition_initiale",
                 f"offset x {offset:g}", x=offset, offset_scale=offset,
                 nominal=est_nominal, reference=est_nominal))


AXES = [
    axe("commandes", "commandes",
        "Familles de commandes jamais vues a l'entrainement",
        "", scenarios_commandes),
    axe("amplitude", "commandes",
        "Extrapolation en amplitude de commande (famille creneaux)",
        "Amplitude de commande A", scenarios_amplitude),
    axe("horizon", "horizon",
        "Horizon plus long que celui vu a l'entrainement",
        "Horizon T (pas)", scenarios_horizon),
    axe("cadence", "capteurs",
        "Cadence du GPS et des mesures de distance",
        "ratio_gps (1 mesure tous les N pas)", scenarios_cadence),
    axe("panne", "capteurs",
        "Panne GPS + distances (seul l'accelerometre subsiste)",
        "Duree de la panne (s)", scenarios_panne),
    axe("aberrations", "capteurs",
        "Mesures aberrantes (bruit x10 sur une fraction des mesures)",
        "Taux de mesures aberrantes", scenarios_aberrations),
    axe("bruit_r", "bruit",
        "Bruit de mesure, y compris hors de la plage d'entrainement",
        r"$1/r^2$ [dB]", scenarios_bruit_r),
    axe("bruit_q", "bruit",
        "Bruit de process (jamais varie a l'entrainement)",
        "Facteur sur l'ecart-type du bruit de process", scenarios_bruit_q),
    axe("geometrie", "geometrie",
        "Geometrie de la formation (regime de non-linearite de h)",
        "", scenarios_geometrie),
    axe("condition_initiale", "condition_initiale",
        "Erreur sur l'etat initial (les filtres partent toujours de x0 nominal)",
        "Amplitude de la perturbation initiale (x chol(P0))",
        scenarios_condition_initiale),
]

# Numerotation des scenarios. La graine de chaque scenario vient de son rang.
# Surtout pas de hash(nom) : Python randomise le hachage des chaines a chaque
# lancement, deux executions ne donneraient pas les memes tirages.
numero = 0
for un_axe in AXES:
    for un_scenario in un_axe["scenarios"]:
        un_scenario["numero"] = numero
        numero += 1


# =========================================================================
# 3. EVALUATION
# =========================================================================

def charge_modeles(sm):
    """Charge {nom_modele: {archi: KalmanNetNN}}, en ignorant les manquants."""
    charges = {}
    for nom_modele in MODELES:
        dossier = MODELES[nom_modele]
        trouves = {}
        for archi in ARCHIS:
            chemin = os.path.join(dossier, f"knet_{archi}.pt")
            if not os.path.exists(chemin):
                print(f"   !! checkpoint absent, ignore : {chemin}")
                continue
            etat = torch.load(chemin, map_location=sm.device)
            model = KalmanNetNN(sm, archi=etat.get("archi", archi))
            model.load_state_dict(etat["state_dict"])
            model.eval()
            trouves[archi] = model
        if len(trouves) > 0:
            charges[nom_modele] = trouves
            print(f"   {nom_modele:16s} : {sorted(trouves)}")
    return charges


def get_systeme(caches, formation, ratio_gps):
    """Renvoie (SystemModel, EKF) pour une geometrie et une cadence donnees.

    Construits une seule fois puis memorises dans caches : les axes geometrie
    et cadence changent le modele systeme, le rebatir a chaque trajectoire
    couterait cher pour rien.
    """
    cle = (formation, ratio_gps)
    if cle not in caches:
        sm = SystemModel(x0=formation, ratio_gps=ratio_gps)
        caches[cle] = (sm, EKF(sm))
    return caches[cle]


def mse(x_estime, x_vrai, composantes, debut=0, fin=None):
    """MSE sur un sous-ensemble de composantes d'etat et de pas de temps."""
    ecart = x_estime[debut:fin, composantes, 0] - x_vrai[debut:fin, composantes, 0]
    return (ecart ** 2).mean().item()


def bornes_des_fenetres(T):
    """(debut, fin) de chaque fenetre temporelle, en indices de pas.

      total       : toute la trajectoire
      transitoire : les 2 premieres secondes -- mesure la reconvergence apres
                    une panne GPS ou une erreur d'etat initial
      etabli      : les 2 dernieres secondes
      bloc_debut  : les 160 premiers pas estimes
      bloc_fin    : les 160 derniers pas estimes
    Le rapport entre bloc_fin et bloc_debut mesure la derive quand T depasse
    l'horizon vu a l'entrainement ; un Delta_dB global la moyennerait. Les deux
    blocs ont exactement la meme longueur, donc leur rapport vaut 1 a T = 160.

    Les quatre sous-fenetres demarrent au pas 1, pas au pas 0 : au pas 0 les
    deux filtres valent x0 nominal, leur erreur est identique, et l'inclure
    ajouterait le meme terme au numerateur et au denominateur du rapport.
    total garde le pas 0 pour rester comparable aux resultats deja obtenus
    avec eval_ood.py.
    """
    n_transitoire = int(round(TRANSITOIRE_S / DT))
    taille_bloc = min(T_REF, T)
    return {"total":       (0, T + 1),
            "transitoire": (1, n_transitoire + 1),
            "etabli":      (T + 1 - n_transitoire, T + 1),
            "bloc_debut":  (1, taille_bloc + 1),
            "bloc_fin":    (T + 1 - taille_bloc, T + 1)}


def accumulateur_vide():
    """Un accumulateur par estimateur : les MSE de chaque trajectoire."""
    acc = {"divergences": 0, "mse_all": []}
    for fenetre in FENETRES:
        acc[fenetre] = []
    return acc


def ajoute_trajectoire(acc, x_estime, x_vrai, fenetres):
    """Range les MSE d'une trajectoire dans un accumulateur."""
    for fenetre in FENETRES:
        debut, fin = fenetres[fenetre]
        acc[fenetre].append(mse(x_estime, x_vrai, IDX_POS_D2, debut, fin))
    acc["mse_all"].append(mse(x_estime, x_vrai, IDX_POS_ALL))


def en_db(mse_knet, mse_ekf):
    """Delta_dB = 10 log10(MSE_KNet / MSE_EKF). None si le calcul n'a pas de sens."""
    if mse_knet <= 0 or mse_ekf <= 0:
        return None
    if not np.isfinite(mse_knet) or not np.isfinite(mse_ekf):
        return None
    return float(10.0 * np.log10(mse_knet / mse_ekf))


def evalue_scenario(caches, modeles, sc, n_mc):
    """Evalue tous les modeles sur n_mc trajectoires d'un scenario."""
    sm, ekf = get_systeme(caches, sc["formation"], sc["ratio_gps"])
    # Les modeles ont ete construits sur un autre SystemModel : il faut les y
    # rattacher, sinon ils continueraient d'utiliser l'ancien x0 et l'ancien h.
    for nom_modele in modeles:
        for archi in modeles[nom_modele]:
            rebind(modeles[nom_modele][archi], sm)

    T = sc["T"]
    fenetres = bornes_des_fenetres(T)
    rng = np.random.default_rng(SEED_EVAL + sc["numero"])

    acc_ekf = accumulateur_vide()
    acc_knet = {}
    for nom_modele in modeles:
        acc_knet[nom_modele] = {}
        for archi in modeles[nom_modele]:
            acc_knet[nom_modele][archi] = accumulateur_vide()

    for _ in range(n_mc):
        u = build_command(T, sm.dt, rng, kind=sc["famille"], A=sc["amplitude"])
        X, Y, U, M = generate_trajectory(sm, rng, u_seq=u, T=T,
                                         r_scale=sc["r_scale"],
                                         q_scale=sc["q_scale"],
                                         offset_scale=sc["offset_scale"])
        if len(sc["pannes"]) > 0 or sc["taux_aberrant"] > 0.0:
            Y, M = corrupt_observations(Y, M, rng, sm,
                                        outages=sc["pannes"],
                                        outlier_rate=sc["taux_aberrant"])

        x_ekf, _ = ekf.run(Y, U, M)
        mse_ekf = mse(x_ekf, X, IDX_POS_D2)
        ajoute_trajectoire(acc_ekf, x_ekf, X, fenetres)

        for nom_modele in modeles:
            for archi in modeles[nom_modele]:
                x_knet = run_knet(sm, modeles[nom_modele][archi], Y, U, M)
                mse_knet = mse(x_knet, X, IDX_POS_D2)
                acc = acc_knet[nom_modele][archi]
                if not np.isfinite(mse_knet):
                    acc["divergences"] += 1
                    continue
                if mse_knet > FACTEUR_DIVERGENCE * mse_ekf:
                    acc["divergences"] += 1
                ajoute_trajectoire(acc, x_knet, X, fenetres)

    resultat = {"libelle": sc["libelle"], "x": sc["x"], "T": T,
                "nominal": sc["nominal"], "reference": sc["reference"],
                "ekf_d2": float(np.mean(acc_ekf["total"])),
                "ekf_all": float(np.mean(acc_ekf["mse_all"])),
                "modeles": {}}
    for nom_modele in modeles:
        resultat["modeles"][nom_modele] = {}
        for archi in modeles[nom_modele]:
            acc = acc_knet[nom_modele][archi]
            if len(acc["total"]) == 0:
                resultat["modeles"][nom_modele][archi] = None
                continue
            entree = {"mse_d2": float(np.mean(acc["total"])),
                      "mse_all": float(np.mean(acc["mse_all"])),
                      "div_rate": acc["divergences"] / n_mc}
            entree["delta_d2"] = en_db(np.mean(acc["total"]),
                                       np.mean(acc_ekf["total"]))
            entree["delta_all"] = en_db(np.mean(acc["mse_all"]),
                                        np.mean(acc_ekf["mse_all"]))
            for fenetre in ("transitoire", "etabli", "bloc_debut", "bloc_fin"):
                entree["delta_" + fenetre] = en_db(np.mean(acc[fenetre]),
                                                   np.mean(acc_ekf[fenetre]))
            resultat["modeles"][nom_modele][archi] = entree
    return resultat


def resume_console(resultat, modeles):
    """Construit la ligne affichee pour un scenario : un bout par estimateur."""
    bouts = []
    for nom_modele in modeles:
        for archi in sorted(modeles[nom_modele]):
            entree = resultat["modeles"][nom_modele][archi]
            etiquette = nom_modele[0] + "-" + archi[-1]
            if entree is None or entree["delta_d2"] is None:
                bouts.append(etiquette + ":    n/a")
                continue
            texte = f"{etiquette}:{entree['delta_d2']:+7.2f}dB"
            if entree["div_rate"] > 0:
                texte += f"({100 * entree['div_rate']:.0f}%div)"
            bouts.append(texte)
    return "  ".join(bouts)


def evalue_axe(caches, modeles, un_axe, n_mc):
    """Evalue tous les scenarios d'un axe."""
    resultats_axe = {"titre": un_axe["titre"], "xlabel": un_axe["xlabel"],
                     "groupe": un_axe["groupe"], "scenarios": {}}
    for sc in un_axe["scenarios"]:
        resultat = evalue_scenario(caches, modeles, sc, n_mc)
        resultats_axe["scenarios"][sc["nom"]] = resultat
        print(f"   {sc['libelle']:44s} EKF={resultat['ekf_d2']:9.3f} | "
              + resume_console(resultat, modeles))
    return resultats_axe


def controle_coherence(resultats, modeles):
    """Les scenarios nominaux decrivent la MEME condition d'entrainement.

    Ils sont estimes sur des trajectoires independantes : leurs Delta_dB
    doivent coincider a la variance Monte-Carlo pres. Une etendue importante
    signale un N_MC trop faible pour conclure, ou un modele si instable que sa
    reference elle-meme n'est pas reproductible.
    """
    print("\n== Controle de coherence des scenarios nominaux ==")
    tout_va_bien = True
    for nom_modele in modeles:
        for archi in sorted(modeles[nom_modele]):
            valeurs = []
            for nom_axe in resultats:
                for nom_sc in resultats[nom_axe]["scenarios"]:
                    sc_res = resultats[nom_axe]["scenarios"][nom_sc]
                    if not sc_res["nominal"]:
                        continue
                    entree = sc_res["modeles"][nom_modele][archi]
                    if entree is not None and entree["delta_d2"] is not None:
                        valeurs.append(entree["delta_d2"])
            if len(valeurs) < 2:
                continue
            etendue = max(valeurs) - min(valeurs)
            if etendue < 3.0:
                statut = "OK "
            else:
                statut = "!! "
                tout_va_bien = False
            print(f"   {statut}{nom_modele} / {archi} : {len(valeurs)} nominaux, "
                  f"Delta_dB de {min(valeurs):+.2f} a {max(valeurs):+.2f} "
                  f"(etendue {etendue:.2f} dB)")
    if not tout_va_bien:
        print("   !! etendue > 3 dB : conditions nominales identiques mais "
              "resultats disperses -> augmenter N_MC, ou modele instable.")
    return tout_va_bien


# =========================================================================
# 4. FIGURES ET TABLEAU
# =========================================================================

def series_du_graphique(resultats_axe, modeles):
    """[(nom_modele, archi, valeurs)] : une serie par courbe ou groupe de barres.

    valeurs suit l'ordre des scenarios de l'axe ; np.nan la ou le modele n'a
    pas de resultat exploitable.
    """
    noms_scenarios = list(resultats_axe["scenarios"])
    series = []
    for nom_modele in modeles:
        for archi in sorted(modeles[nom_modele]):
            valeurs = []
            for nom_sc in noms_scenarios:
                entree = resultats_axe["scenarios"][nom_sc]["modeles"][nom_modele][archi]
                if entree is None or entree["delta_d2"] is None:
                    valeurs.append(np.nan)
                else:
                    valeurs.append(entree["delta_d2"])
            series.append((nom_modele, archi, np.array(valeurs)))
    return series


def figure_axe(resultats_axe, un_axe, modeles, outdir):
    """Une figure par axe : courbes si l'axe est numerique, barres sinon."""
    scenarios_res = resultats_axe["scenarios"]
    noms_scenarios = list(scenarios_res)
    series = series_du_graphique(resultats_axe, modeles)

    largeur = max(8.5, 1.6 * len(noms_scenarios) + 3)
    fig, ax = plt.subplots(figsize=(largeur, 5.5))

    if un_axe["xlabel"] != "":
        abscisses = []
        for nom_sc in noms_scenarios:
            abscisses.append(scenarios_res[nom_sc]["x"])
        for nom_modele, archi, valeurs in series:
            ax.plot(abscisses, valeurs, "o" + STYLES[archi], lw=2, ms=5,
                    color=COULEURS.get((nom_modele, archi)),
                    label=f"{nom_modele} / {archi}")
        ax.set_xlabel(un_axe["xlabel"])
        for nom_sc in noms_scenarios:
            if not scenarios_res[nom_sc]["reference"]:
                continue
            # Sur l'axe amplitude la reference n'est pas une condition
            # d'entrainement (la famille creneaux n'a jamais ete vue) : on ne
            # lui met donc pas la meme etiquette.
            if scenarios_res[nom_sc]["nominal"]:
                etiquette = " entrainement"
            else:
                etiquette = " reference de l'axe"
            ax.axvline(scenarios_res[nom_sc]["x"], color="grey",
                       ls=":", lw=1.5)
            ax.text(scenarios_res[nom_sc]["x"], ax.get_ylim()[1],
                    etiquette, fontsize=8, ha="left", va="top",
                    color="dimgrey")
    else:
        abscisses = np.arange(len(noms_scenarios))
        largeur_barre = 0.8 / max(len(series), 1)
        for i in range(len(series)):
            nom_modele, archi, valeurs = series[i]
            decalage = i * largeur_barre - 0.4 + largeur_barre / 2
            ax.bar(abscisses + decalage, valeurs, largeur_barre,
                   color=COULEURS.get((nom_modele, archi)),
                   label=f"{nom_modele} / {archi}")
        etiquettes = []
        for nom_sc in noms_scenarios:
            etiquettes.append(scenarios_res[nom_sc]["libelle"].replace(" (", "\n("))
        ax.set_xticks(abscisses)
        ax.set_xticklabels(etiquettes, fontsize=9)

    ax.axhline(0, color="k", lw=1.2)
    ax.set_ylabel(r"$\Delta_{dB} = 10\log_{10}(MSE_{KNet}/MSE_{EKF})$")
    ax.set_title(f"{un_axe['titre']}\nen dessous de 0 : KalmanNet bat l'EKF",
                 fontsize=12)
    ax.grid(True, axis="y", ls=":", alpha=.7)
    ax.legend(fontsize=9)
    fig.tight_layout()
    chemin = os.path.join(outdir, f"axe_{un_axe['nom']}.png")
    fig.savefig(chemin, dpi=130)
    plt.close(fig)
    return chemin


def figure_synthese(resultats, modeles, outdir):
    """Pire Delta_dB de chaque groupe : la vue d'ensemble de la degradation."""
    series = []
    for nom_modele in modeles:
        for archi in sorted(modeles[nom_modele]):
            pires = []
            for groupe in GROUPES:
                valeurs = []
                for nom_axe in resultats:
                    if resultats[nom_axe]["groupe"] != groupe:
                        continue
                    for nom_sc in resultats[nom_axe]["scenarios"]:
                        entree = resultats[nom_axe]["scenarios"][nom_sc]["modeles"][nom_modele][archi]
                        if entree is not None and entree["delta_d2"] is not None:
                            valeurs.append(entree["delta_d2"])
                if len(valeurs) > 0:
                    pires.append(max(valeurs))
                else:
                    pires.append(np.nan)
            series.append((nom_modele, archi, pires))

    abscisses = np.arange(len(GROUPES))
    largeur_barre = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i in range(len(series)):
        nom_modele, archi, pires = series[i]
        decalage = i * largeur_barre - 0.4 + largeur_barre / 2
        ax.bar(abscisses + decalage, pires, largeur_barre,
               color=COULEURS.get((nom_modele, archi)),
               label=f"{nom_modele} / {archi}")
    etiquettes = []
    for groupe in GROUPES:
        etiquettes.append(groupe.replace("_", "\n"))
    ax.axhline(0, color="k", lw=1.2)
    ax.set_xticks(abscisses)
    ax.set_xticklabels(etiquettes, fontsize=9)
    ax.set_ylabel(r"pire $\Delta_{dB}$ de l'axe")
    ax.set_title("Synthese : degradation dans le cas le plus defavorable "
                 "de chaque axe", fontsize=12)
    ax.grid(True, axis="y", ls=":", alpha=.7)
    ax.legend(fontsize=9)
    fig.tight_layout()
    chemin = os.path.join(outdir, "synthese.png")
    fig.savefig(chemin, dpi=130)
    plt.close(fig)
    return chemin


def figure_temporelle(caches, modeles, sc, outdir, seed=999):
    """Erreur de position du drone 2 au cours du temps, sur une trajectoire."""
    sm, ekf = get_systeme(caches, sc["formation"], sc["ratio_gps"])
    for nom_modele in modeles:
        for archi in modeles[nom_modele]:
            rebind(modeles[nom_modele][archi], sm)

    T = sc["T"]
    rng = np.random.default_rng(seed)
    u = build_command(T, sm.dt, rng, kind=sc["famille"], A=sc["amplitude"])
    X, Y, U, M = generate_trajectory(sm, rng, u_seq=u, T=T,
                                     r_scale=sc["r_scale"],
                                     q_scale=sc["q_scale"],
                                     offset_scale=sc["offset_scale"])
    if len(sc["pannes"]) > 0 or sc["taux_aberrant"] > 0.0:
        Y, M = corrupt_observations(Y, M, rng, sm,
                                    outages=sc["pannes"],
                                    outlier_rate=sc["taux_aberrant"])
    temps = np.arange(T + 1) * sm.dt

    def erreur_position(x_estime):
        # float64 : sur un modele divergent la norme deborde en float32.
        ecart = (x_estime[:, 8:10, 0] - X[:, 8:10, 0]).cpu().numpy()
        return np.linalg.norm(ecart.astype(np.float64), axis=1)

    x_ekf, _ = ekf.run(Y, U, M)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(temps, erreur_position(x_ekf), color="green", lw=1.6, label="EKF")
    for nom_modele in modeles:
        for archi in sorted(modeles[nom_modele]):
            x_knet = run_knet(sm, modeles[nom_modele][archi], Y, U, M)
            ax.plot(temps, erreur_position(x_knet), lw=1.5, ls=STYLES[archi],
                    color=COULEURS.get((nom_modele, archi)),
                    label=f"{nom_modele} / {archi}")
    for k_debut, k_fin, _capteurs in sc["pannes"]:
        ax.axvspan(k_debut * sm.dt, k_fin * sm.dt, color="red", alpha=.12)
        ax.text((k_debut + k_fin) / 2 * sm.dt, ax.get_ylim()[1], "panne GPS",
                fontsize=8, ha="center", va="top", color="firebrick")

    ax.set_yscale("log")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Erreur de position drone 2 (m)")
    ax.set_title(f"Reponse temporelle - {sc['libelle']}", fontsize=12)
    ax.grid(True, ls=":", alpha=.7)
    ax.legend(fontsize=9)
    fig.tight_layout()
    chemin = os.path.join(outdir, f"temporel_{sc['nom']}.png")
    fig.savefig(chemin, dpi=130)
    plt.close(fig)
    return chemin


def ecrit_tableau(resultats, modeles, outdir, n_mc):
    """Ecrit tableau.md : les Delta_dB, prets a coller dans un rapport."""
    colonnes = []
    for nom_modele in modeles:
        for archi in sorted(modeles[nom_modele]):
            colonnes.append((nom_modele, archi))

    lignes = ["# Cartographie de la generalisation - Delta_dB",
              "",
              "`Delta_dB = 10 log10(MSE_KNet / MSE_EKF)` sur la position du "
              "drone 2. **Negatif = KalmanNet bat l'EKF.**",
              f"Comparaison appariee, {n_mc} trajectoires par scenario. "
              "`div` = part des runs ou MSE_KNet depasse 100x celle de l'EKF.",
              ""]
    for groupe in GROUPES:
        lignes.append(f"## {groupe.replace('_', ' ')}")
        lignes.append("")
        for nom_axe in resultats:
            resultats_axe = resultats[nom_axe]
            if resultats_axe["groupe"] != groupe:
                continue
            entetes = []
            for nom_modele, archi in colonnes:
                entetes.append(f"{nom_modele} / {archi}")
            lignes.append(f"### {resultats_axe['titre']}")
            lignes.append("")
            lignes.append("| scenario | MSE EKF | " + " | ".join(entetes) + " |")
            lignes.append("|---|---|" + "---|" * len(colonnes))
            for nom_sc in resultats_axe["scenarios"]:
                sc_res = resultats_axe["scenarios"][nom_sc]
                cellules = []
                for nom_modele, archi in colonnes:
                    entree = sc_res["modeles"][nom_modele][archi]
                    if entree is None or entree["delta_d2"] is None:
                        cellules.append("n/a")
                        continue
                    texte = f"{entree['delta_d2']:+.2f}"
                    if entree["div_rate"] > 0:
                        texte += f" ({100 * entree['div_rate']:.0f}% div)"
                    cellules.append(texte)
                marque = ""
                if sc_res["reference"]:
                    marque = " *(ref)*"
                lignes.append(f"| {sc_res['libelle']}{marque} | "
                              f"{sc_res['ekf_d2']:.3f} | "
                              + " | ".join(cellules) + " |")
            lignes.append("")

    chemin = os.path.join(outdir, "tableau.md")
    with open(chemin, "w", encoding="utf-8") as fichier:
        fichier.write("\n".join(lignes))
    return chemin


# =========================================================================
# 5. MAIN
# =========================================================================

def main(n_mc=N_MC, outdir=OUT_DIR, noms_axes=None):
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(CFG.SEED)
    np.random.seed(CFG.SEED)

    # L'evaluation impose toujours ses propres commandes via u_seq ; on
    # neutralise le flag pour qu'aucun appel indirect ne rebascule dessus.
    CFG.TRAIN_CMD_RANDOMIZE = False

    caches = {}
    sm_ref, _ = get_systeme(caches, "triangle", 5)
    print(f">> Device : {sm_ref.device}")
    print("== Chargement des modeles ==")
    modeles = charge_modeles(sm_ref)
    if len(modeles) == 0:
        raise SystemExit(
            "Aucun checkpoint trouve. Lancez d'abord : python train_models.py")

    axes_a_faire = []
    for un_axe in AXES:
        if noms_axes is None or un_axe["nom"] in noms_axes:
            axes_a_faire.append(un_axe)

    resultats = {}
    for un_axe in axes_a_faire:
        print(f"\n== Axe : {un_axe['titre']} ==")
        resultats[un_axe["nom"]] = evalue_axe(caches, modeles, un_axe, n_mc)

    produits = []
    for un_axe in axes_a_faire:
        produits.append(figure_axe(resultats[un_axe["nom"]], un_axe,
                                   modeles, outdir))

    # Series temporelles la ou la dynamique compte plus que la moyenne : on
    # prend le dernier scenario de l'axe, c'est-a-dire le cas le plus severe.
    for un_axe in axes_a_faire:
        if un_axe["nom"] in ("panne", "horizon", "condition_initiale"):
            produits.append(figure_temporelle(caches, modeles,
                                              un_axe["scenarios"][-1], outdir))

    produits.append(figure_synthese(resultats, modeles, outdir))
    produits.append(ecrit_tableau(resultats, modeles, outdir, n_mc))

    chemin_json = os.path.join(outdir, "resultats.json")
    modeles_charges = {}
    for nom_modele in modeles:
        modeles_charges[nom_modele] = sorted(modeles[nom_modele])
    with open(chemin_json, "w", encoding="utf-8") as fichier:
        json.dump({"meta": {"n_mc": n_mc, "seed": SEED_EVAL,
                            "modeles": modeles_charges,
                            "facteur_divergence": FACTEUR_DIVERGENCE},
                   "axes": resultats}, fichier, indent=2)
    produits.append(chemin_json)

    controle_coherence(resultats, modeles)

    print("\n== Sorties ==")
    for chemin in produits:
        print("  ", chemin)
    return resultats


if __name__ == "__main__":
    if "--fumee" in sys.argv:
        main(n_mc=2, outdir=os.path.join(OUT_DIR, "fumee"))
    else:
        main()
