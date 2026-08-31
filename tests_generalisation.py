"""
Tests de l'outillage de l'etude de generalisation.

Pas besoin de pytest : un simple `python tests_generalisation.py`.

Le premier test est le plus important. Il compare la generation de
trajectoires entre la version actuelle et KalmanNet_Drones_origine.py (copie
figee d'avant modification) : tant qu'il passe, les checkpoints entraines
avant l'ajout des parametres de l'etude restent valides.
"""

import sys

import numpy as np
import torch

import KalmanNet_Drones as ACTUEL
import KalmanNet_Drones_origine as ORIGINE
import etude_generalisation as ETUDE

from KalmanNet_Drones import (CFG, SystemModel, KalmanNetNN, make_formation,
                              generate_trajectory, corrupt_observations,
                              run_knet, rebind, CAPTEURS_GPS, CAPTEURS_IMU)

echecs = []


def verifie(condition, message):
    """Enregistre le resultat d'une verification et l'affiche."""
    if condition:
        print("  OK  " + message)
    else:
        print("  ECHEC  " + message)
        echecs.append(message)


def trajectoire(module, r_scale=1.0, init_perturb=True):
    """Genere une trajectoire avec des graines fixees, pour comparaison."""
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    sm = module.SystemModel()
    return module.generate_trajectory(sm, rng, r_scale=r_scale,
                                      init_perturb=init_perturb)


def memes_trajectoires(a, b):
    """Vrai si deux trajectoires (X, Y, U, M) sont identiques bit a bit."""
    for i in range(len(a)):
        if a[i].shape != b[i].shape:
            return False
        if not torch.equal(a[i], b[i]):
            return False
    return True


# ------------------------------------------------------- 1. non-regression
def test_non_regression():
    print("\n== 1. La generation de trajectoires n'a pas change ==")
    for randomise in (True, False):
        for offset_p0 in (True, False):
            ORIGINE.CFG.TRAIN_CMD_RANDOMIZE = randomise
            ACTUEL.CFG.TRAIN_CMD_RANDOMIZE = randomise
            ORIGINE.CFG.INIT_OFFSET_P0 = offset_p0
            ACTUEL.CFG.INIT_OFFSET_P0 = offset_p0
            reglage = f"commandes_randomisees={randomise}, offset_P0={offset_p0}"

            verifie(memes_trajectoires(trajectoire(ORIGINE),
                                       trajectoire(ACTUEL)),
                    reglage)
            verifie(memes_trajectoires(trajectoire(ORIGINE, r_scale=0.3),
                                       trajectoire(ACTUEL, r_scale=0.3)),
                    reglage + ", r_scale=0.3")
            verifie(memes_trajectoires(trajectoire(ORIGINE, init_perturb=False),
                                       trajectoire(ACTUEL, init_perturb=False)),
                    reglage + ", sans perturbation initiale")

    ORIGINE.CFG.TRAIN_CMD_RANDOMIZE = ACTUEL.CFG.TRAIN_CMD_RANDOMIZE = True
    ORIGINE.CFG.INIT_OFFSET_P0 = ACTUEL.CFG.INIT_OFFSET_P0 = True

    # Les nouveaux parametres, a leur valeur par defaut, ne changent rien.
    reference = trajectoire(ORIGINE)

    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    avec_defauts = generate_trajectory(SystemModel(), rng, T=None,
                                       q_scale=1.0, offset_scale=None)
    verifie(memes_trajectoires(reference, avec_defauts),
            "T=None, q_scale=1.0, offset_scale=None ne changent rien")

    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    avec_explicites = generate_trajectory(SystemModel(), rng, T=CFG.T,
                                          q_scale=1.0,
                                          offset_scale=CFG.INIT_OFFSET_SCALE)
    verifie(memes_trajectoires(reference, avec_explicites),
            "les memes valeurs passees explicitement ne changent rien")


# ------------------------------------------------------------ 2. formations
def distance(positions, i, j):
    return float(np.linalg.norm(positions[i] - positions[j]))


def test_formations():
    print("\n== 2. Geometries de formation ==")
    sm_defaut = SystemModel()
    verifie(torch.equal(sm_defaut.x0, SystemModel(x0="triangle").x0),
            "make_formation('triangle') reproduit le x0 d'origine")
    verifie(make_formation("triangle").shape == (24,),
            "un x0 a bien 24 composantes")

    positions = {}
    for forme in ETUDE.LIBELLES_FORMATION:
        # .cpu() avant .numpy() : indispensable sur GPU, sans effet en CPU.
        positions[forme] = SystemModel(x0=forme).x0.reshape(3, 8)[:, :2].cpu().numpy()
        p = positions[forme]
        print(f"      {forme:9s} d12={distance(p, 0, 1):6.2f} "
              f"d23={distance(p, 1, 2):6.2f} d13={distance(p, 0, 2):6.2f}")

    p = positions["ligne"]
    ecart = abs(distance(p, 0, 1) + distance(p, 1, 2) - distance(p, 0, 2))
    verifie(ecart < 1e-5,
            "la formation 'ligne' est bien colineaire (d12 + d23 = d13)")
    verifie(distance(positions["serree"], 0, 1)
            < distance(positions["triangle"], 0, 1)
            < distance(positions["large"], 0, 1),
            "les distances croissent : serree < triangle < large")

    for mauvais_x0 in (np.zeros(23), "inconnue"):
        try:
            SystemModel(x0=mauvais_x0)
            verifie(False, f"x0={mauvais_x0!r} aurait du etre refuse")
        except ValueError:
            verifie(True, f"x0 invalide refuse ({type(mauvais_x0).__name__})")


# ------------------------------------------------------- 3. cadence capteurs
def test_ratio_gps():
    print("\n== 3. Cadence des capteurs ==")
    for ratio in (1, 2, 5, 10, 20):
        sm = SystemModel(ratio_gps=ratio)
        n_gps = 0
        n_imu = 0
        for k in range(1, 41):
            masque = sm.obs_mask(k)
            n_gps += int(masque[0].item())
            n_imu += int(masque[2].item())
        verifie(n_gps == 40 // ratio and n_imu == 40,
                f"ratio_gps={ratio:2d} -> {n_gps}/40 GPS, {n_imu}/40 IMU")


# ------------------------------------------- 4. corruption des observations
def test_corrupt_observations():
    print("\n== 4. corrupt_observations ==")
    sm = SystemModel()
    rng = np.random.default_rng(3)
    X, Y, U, M = generate_trajectory(sm, rng)
    Y_initial = Y.clone()
    M_initial = M.clone()

    Yc, Mc = corrupt_observations(Y, M, rng, sm)
    verifie(torch.equal(Yc, Y_initial) and torch.equal(Mc, M_initial),
            "sans degradation demandee, les mesures sont inchangees")

    Yc, Mc = corrupt_observations(Y, M, rng, sm,
                                  outages=((30, 59, CAPTEURS_GPS),))
    verifie(torch.equal(Y, Y_initial) and torch.equal(M, M_initial),
            "les entrees ne sont jamais modifiees en place")
    verifie(float(Mc[30:60, CAPTEURS_GPS].sum()) == 0.0,
            "la fenetre de panne annule le masque GPS (bornes incluses)")
    verifie(float(Mc[30:60, CAPTEURS_IMU].sum()) > 0.0,
            "l'IMU reste disponible pendant la panne GPS")
    verifie(torch.equal(Mc[:30], M_initial[:30])
            and torch.equal(Mc[60:], M_initial[60:]),
            "hors de la fenetre, le masque est intact")
    verifie(torch.equal(Yc, Y_initial),
            "une panne ne modifie pas Y, seulement le masque")

    Yc, Mc = corrupt_observations(Y, M, rng, sm, outlier_rate=1.0,
                                  outlier_scale=50.0)
    ecart = (Yc - Y_initial).abs().squeeze(-1)
    verifie(torch.equal(Mc, M_initial),
            "les mesures aberrantes ne touchent pas au masque")
    verifie(float(ecart[M_initial == 0].max()) == 0.0,
            "aucune mesure indisponible n'est corrompue")
    verifie(float(ecart[M_initial > 0].mean()) > 0.0,
            "les mesures disponibles sont bien corrompues")


# ---------------------------------------------------------------- 5. rebind
def test_rebind():
    print("\n== 5. rebind sur un nouveau SystemModel ==")
    sm_reference = SystemModel()
    model = KalmanNetNN(sm_reference, archi="archi2")
    model.eval()

    sm_ligne = SystemModel(x0="ligne")
    rebind(model, sm_ligne)
    verifie(model.sm is sm_ligne and model.h == sm_ligne.h
            and model.f == sm_ligne.f,
            "f, h et sm pointent vers le nouveau modele systeme")
    verifie(torch.equal(model.prior_Q, sm_ligne.prior_Q)
            and torch.equal(model.prior_Sigma, sm_ligne.prior_Sigma)
            and torch.equal(model.prior_S, sm_ligne.prior_S),
            "les priors qui initialisent les GRU sont ceux du nouveau systeme")
    verifie(torch.equal(sm_ligne.Q, sm_reference.Q)
            and torch.equal(sm_ligne.P0, sm_reference.P0)
            and torch.equal(sm_ligne.R, sm_reference.R),
            "changer la geometrie ne modifie ni Q, ni P0, ni R")

    rng = np.random.default_rng(11)
    X, Y, U, M = generate_trajectory(sm_ligne, rng)
    x_estime = run_knet(sm_ligne, model, Y, U, M)
    verifie(bool(torch.isfinite(x_estime).all())
            and x_estime.shape == X.shape,
            "une passe run_knet apres rebind donne un resultat fini")


# ------------------------------------------------------------- 6. scenarios
def test_scenarios():
    print("\n== 6. Coherence des scenarios ==")
    noms_vus = []
    nominaux = []
    for un_axe in ETUDE.AXES:
        references = []
        for sc in un_axe["scenarios"]:
            if sc["reference"]:
                references.append(sc)
            if sc["nominal"]:
                nominaux.append(sc)
            noms_vus.append(sc["nom"])
            if sc["axe"] != un_axe["nom"]:
                verifie(False, f"{sc['nom']} annonce l'axe {sc['axe']} "
                               f"au lieu de {un_axe['nom']}")
        verifie(len(references) == 1,
                f"axe '{un_axe['nom']}' : exactement une reference "
                f"({len(references)} trouvee(s))")

    verifie(len(noms_vus) == len(set(noms_vus)),
            f"{len(noms_vus)} scenarios, tous de nom distinct")

    groupes_utilises = []
    for un_axe in ETUDE.AXES:
        if un_axe["groupe"] not in groupes_utilises:
            groupes_utilises.append(un_axe["groupe"])
    verifie(sorted(groupes_utilises) == sorted(ETUDE.GROUPES),
            "les groupes declares couvrent exactement les axes")

    # Garantie structurelle, et non statistique : tous les scenarios nominaux
    # decrivent la meme condition. C'est ce qui rend les axes comparables
    # entre eux et donne un sens au controle de coherence de l'evaluation.
    reglages_nominaux = []
    for sc in nominaux:
        reglages = (sc["famille"], sc["amplitude"], sc["T"], sc["r_scale"],
                    sc["q_scale"], sc["offset_scale"], sc["formation"],
                    sc["ratio_gps"], tuple(sc["pannes"]), sc["taux_aberrant"])
        if reglages not in reglages_nominaux:
            reglages_nominaux.append(reglages)
    verifie(len(reglages_nominaux) == 1,
            f"les {len(nominaux)} scenarios nominaux decrivent une condition "
            f"identique ({len(reglages_nominaux)} configuration(s) distincte(s))")

    # Les numeros servent de graines : ils doivent etre uniques et stables.
    numeros = []
    for un_axe in ETUDE.AXES:
        for sc in un_axe["scenarios"]:
            numeros.append(sc["numero"])
    verifie(sorted(numeros) == list(range(len(numeros))),
            f"les numeros de scenario vont de 0 a {len(numeros) - 1} sans trou")

    # Chaque scenario doit etre executable tel quel.
    for un_axe in ETUDE.AXES:
        for sc in un_axe["scenarios"]:
            sm = SystemModel(x0=sc["formation"], ratio_gps=sc["ratio_gps"])
            rng = np.random.default_rng(5)
            u = ETUDE.build_command(sc["T"], sm.dt, rng,
                                    kind=sc["famille"], A=sc["amplitude"])
            X, Y, U, M = generate_trajectory(sm, rng, u_seq=u, T=sc["T"],
                                             r_scale=sc["r_scale"],
                                             q_scale=sc["q_scale"],
                                             offset_scale=sc["offset_scale"])
            if len(sc["pannes"]) > 0 or sc["taux_aberrant"] > 0.0:
                Y, M = corrupt_observations(Y, M, rng, sm,
                                            outages=sc["pannes"],
                                            outlier_rate=sc["taux_aberrant"])
            assert X.shape == (sc["T"] + 1, 24, 1), (sc["nom"], X.shape)
            assert torch.isfinite(X).all(), sc["nom"]
            assert torch.isfinite(Y).all(), sc["nom"]
        verifie(True, f"axe '{un_axe['nom']}' : "
                      f"{len(un_axe['scenarios'])} scenarios executables")


if __name__ == "__main__":
    print(f">> Device : {CFG.DEVICE}  |  CUDA disponible : {torch.cuda.is_available()}")
    CFG.TRAIN_CMD_RANDOMIZE = False   # l'evaluation impose ses commandes via u_seq
    test_non_regression()
    test_formations()
    test_ratio_gps()
    test_corrupt_observations()
    test_rebind()
    test_scenarios()

    print("\n" + "=" * 60)
    if len(echecs) > 0:
        print(f"{len(echecs)} ECHEC(S) :")
        for message in echecs:
            print("  -", message)
        sys.exit(1)
    print("TOUS LES TESTS PASSENT")
