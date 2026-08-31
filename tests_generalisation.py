"""
Tests de l'outillage de generalisation.

Pas de dependance a pytest : un simple `python tests_generalisation.py`.
Le test le plus important est le premier -- il garantit que la parametrisation
de generate_trajectory n'a rien change au comportement d'origine, donc que les
checkpoints entraines avant cette modification restent valides.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

import KalmanNet_Drones as K
from KalmanNet_Drones import (CFG, SystemModel, KalmanNetNN, make_formation,
                              generate_trajectory, corrupt_observations,
                              run_knet, rebind, CAPTEURS_GPS, CAPTEURS_IMU)
import scenarios

# Commit de reference : dernier etat du fichier avant la parametrisation.
COMMIT_BASE = "709a88b"

_echecs = []


def verifie(condition, message):
    if condition:
        print(f"  OK  {message}")
    else:
        print(f"  ECHEC  {message}")
        _echecs.append(message)


# ------------------------------------------------------- 1. non-regression
def test_non_regression():
    """generate_trajectory doit rester bit a bit identique a la version d'origine."""
    print("\n== 1. Non-regression bit a bit de generate_trajectory ==")
    try:
        src = subprocess.check_output(
            ["git", "show", f"{COMMIT_BASE}:KalmanNet_Drones.py"],
            stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  IGNORE  commit {COMMIT_BASE} inaccessible")
        return

    rep = tempfile.mkdtemp()
    chemin = os.path.join(rep, "_kn_base.py")
    with open(chemin, "wb") as fh:
        fh.write(src)
    sys.path.insert(0, rep)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib
    O = importlib.import_module("_kn_base")
    sys.path.pop(0)
    sys.path.remove(rep)

    def trajectoire(mod, **kw):
        torch.manual_seed(0)
        rng = np.random.default_rng(1)
        return mod.generate_trajectory(mod.SystemModel(), rng, **kw)

    for rand in (True, False):
        for p0 in (True, False):
            O.CFG.TRAIN_CMD_RANDOMIZE = CFG.TRAIN_CMD_RANDOMIZE = rand
            O.CFG.INIT_OFFSET_P0 = CFG.INIT_OFFSET_P0 = p0
            for etiquette, kw in (("defauts", {}),
                                  ("r_scale=0.3", {"r_scale": 0.3}),
                                  ("sans perturbation", {"init_perturb": False})):
                a = trajectoire(O, **kw)
                b = trajectoire(K, **kw)
                verifie(all(torch.equal(x, y) for x, y in zip(a, b)),
                        f"rand={rand}, P0={p0}, {etiquette}")
    O.CFG.TRAIN_CMD_RANDOMIZE = CFG.TRAIN_CMD_RANDOMIZE = True
    O.CFG.INIT_OFFSET_P0 = CFG.INIT_OFFSET_P0 = True

    a = trajectoire(O)
    b = trajectoire(K, T=None, q_scale=1.0, offset_scale=None)
    verifie(all(torch.equal(x, y) for x, y in zip(a, b)),
            "les nouveaux kwargs a leur valeur par defaut ne changent rien")
    b = trajectoire(K, T=CFG.T, q_scale=1.0, offset_scale=CFG.INIT_OFFSET_SCALE)
    verifie(all(torch.equal(x, y) for x, y in zip(a, b)),
            "les memes valeurs passees explicitement ne changent rien")


# ------------------------------------------------------------ 2. formations
def test_formations():
    print("\n== 2. Geometries de formation ==")
    sm_defaut = SystemModel()
    verifie(torch.equal(sm_defaut.x0, SystemModel(x0="triangle").x0),
            "make_formation('triangle') reproduit le x0 d'origine")
    verifie(make_formation("triangle").shape == (24,), "x0 a bien 24 composantes")

    pos = {g: SystemModel(x0=g).x0.reshape(3, 8)[:, :2].numpy()
           for g in scenarios.GEOMETRIES}
    d = lambda p, i, j: float(np.linalg.norm(p[i] - p[j]))
    for g, p in pos.items():
        d12, d23, d13 = d(p, 0, 1), d(p, 1, 2), d(p, 0, 2)
        print(f"      {g:9s} d12={d12:6.2f} d23={d23:6.2f} d13={d13:6.2f}")
    p = pos["ligne"]
    verifie(abs(d(p, 0, 1) + d(p, 1, 2) - d(p, 0, 2)) < 1e-5,
            "formation 'ligne' est bien colineaire (d12 + d23 = d13)")
    verifie(d(pos["serree"], 0, 1) < d(pos["triangle"], 0, 1) < d(pos["large"], 0, 1),
            "les distances croissent : serree < triangle < large")

    for x0_invalide in (np.zeros(23), "inconnue"):
        try:
            SystemModel(x0=x0_invalide)
            verifie(False, f"x0={x0_invalide!r} aurait du etre rejete")
        except ValueError:
            verifie(True, f"x0 invalide rejete ({type(x0_invalide).__name__})")


# ------------------------------------------------------- 3. cadence capteurs
def test_ratio_gps():
    print("\n== 3. Cadence des capteurs ==")
    for r in (1, 2, 5, 10, 20):
        sm = SystemModel(ratio_gps=r)
        gps = sum(int(sm.obs_mask(k)[0].item()) for k in range(1, 41))
        imu = sum(int(sm.obs_mask(k)[2].item()) for k in range(1, 41))
        verifie(gps == 40 // r and imu == 40,
                f"ratio_gps={r:2d} -> {gps}/40 GPS, {imu}/40 IMU")


# ------------------------------------------- 4. corruption des observations
def test_corrupt_observations():
    print("\n== 4. corrupt_observations ==")
    sm = SystemModel()
    rng = np.random.default_rng(3)
    X, Y, U, M = generate_trajectory(sm, rng)
    Y0, M0 = Y.clone(), M.clone()

    Yc, Mc = corrupt_observations(Y, M, rng, sm)
    verifie(torch.equal(Yc, Y0) and torch.equal(Mc, M0),
            "sans argument de degradation, les mesures sont inchangees")

    Yc, Mc = corrupt_observations(Y, M, rng, sm,
                                  outages=((30, 59, CAPTEURS_GPS),))
    verifie(torch.equal(Y, Y0) and torch.equal(M, M0),
            "les entrees ne sont jamais mutees")
    verifie(float(Mc[30:60, CAPTEURS_GPS].sum()) == 0.0,
            "la fenetre de panne annule bien le masque GPS (bornes incluses)")
    verifie(float(Mc[30:60, CAPTEURS_IMU].sum()) > 0.0,
            "l'IMU reste disponible pendant la panne GPS")
    verifie(torch.equal(Mc[:30], M0[:30]) and torch.equal(Mc[60:], M0[60:]),
            "hors de la fenetre, le masque est intact")
    verifie(torch.equal(Yc, Y0), "une panne ne modifie pas Y, seulement le masque")

    Yc, Mc = corrupt_observations(Y, M, rng, sm, outlier_rate=1.0,
                                  outlier_scale=50.0)
    ecart = (Yc - Y0).abs().squeeze(-1)
    verifie(torch.equal(Mc, M0), "les aberrations ne touchent pas au masque")
    verifie(float(ecart[M0 == 0].max()) == 0.0,
            "aucune mesure indisponible n'est corrompue")
    verifie(float(ecart[M0 > 0].mean()) > 0.0,
            "les mesures disponibles sont bien corrompues")


# ---------------------------------------------------------------- 5. rebind
def test_rebind():
    print("\n== 5. rebind sur un nouveau SystemModel ==")
    sm_ref = SystemModel()
    model = KalmanNetNN(sm_ref, archi="archi2")
    model.eval()

    sm_geo = SystemModel(x0="ligne")
    rebind(model, sm_geo)
    verifie(model.sm is sm_geo and model.h == sm_geo.h and model.f == sm_geo.f,
            "f, h et sm pointent vers le nouveau modele systeme")
    verifie(torch.equal(model.prior_Q, sm_geo.prior_Q)
            and torch.equal(model.prior_Sigma, sm_geo.prior_Sigma)
            and torch.equal(model.prior_S, sm_geo.prior_S),
            "les priors qui initialisent les GRU sont ceux du nouveau systeme")
    verifie(torch.equal(sm_geo.Q, sm_ref.Q) and torch.equal(sm_geo.P0, sm_ref.P0)
            and torch.equal(sm_geo.R, sm_ref.R),
            "changer la geometrie ne modifie ni Q, ni P0, ni R")

    rng = np.random.default_rng(11)
    X, Y, U, M = generate_trajectory(sm_geo, rng)
    xh = run_knet(sm_geo, model, Y, U, M)
    verifie(bool(torch.isfinite(xh).all()) and xh.shape == X.shape,
            "une passe run_knet apres rebind donne un resultat fini")


# ------------------------------------------------------------- 6. scenarios
def test_scenarios():
    print("\n== 6. Coherence des scenarios ==")
    noms = set()
    for a in scenarios.AXES:
        refs = [s for s in a.scenarios if s.in_distribution]
        verifie(len(refs) == 1,
                f"axe '{a.name}' : exactement une reference ({len(refs)} trouvee(s))")
        for s in a.scenarios:
            verifie(s.name not in noms, f"nom unique : {s.name}") if s.name in noms \
                else noms.add(s.name)
            if s.axis != a.name:
                verifie(False, f"{s.name} declare l'axe {s.axis} au lieu de {a.name}")
    verifie(len(noms) == sum(len(a.scenarios) for a in scenarios.AXES),
            f"{len(noms)} scenarios, tous de nom distinct")
    verifie(set(a.groupe for a in scenarios.AXES) == set(scenarios.GROUPES),
            "les groupes declares couvrent exactement les axes")

    # Garantie structurelle, et non statistique : tous les scenarios nominaux
    # decrivent la meme condition. C'est ce qui rend comparables les axes entre
    # eux, et ce qui donne un sens au controle de coherence de l'evaluation.
    nominaux = [s for a in scenarios.AXES for s in a.scenarios if s.nominal]
    empreintes = {(tuple(sorted(s.sm_kwargs.items())),
                   tuple(sorted(s.gen_kwargs.items())),
                   tuple(sorted(s.cmd.items())),
                   tuple(sorted(s.corrupt.items()))) for s in nominaux}
    verifie(len(empreintes) == 1,
            f"les {len(nominaux)} scenarios nominaux decrivent une condition "
            f"identique ({len(empreintes)} configuration(s) distincte(s))")
    verifie(all(s.nominal for a in scenarios.AXES for s in a.scenarios
                if s.in_distribution and a.name != "amplitude"),
            "seul l'axe amplitude a une reference non nominale")

    # Chaque scenario doit etre executable tel quel.
    from ood_commands import build_command
    for a in scenarios.AXES:
        for s in a.scenarios:
            sm = SystemModel(**s.sm_kwargs)
            rng = np.random.default_rng(5)
            T = int(s.gen_kwargs.get("T", CFG.T))
            u = build_command(T, sm.dt, rng, **s.cmd)
            X, Y, U, M = generate_trajectory(sm, rng, u_seq=u, **s.gen_kwargs)
            if s.corrupt:
                Y, M = corrupt_observations(Y, M, rng, sm, **s.corrupt)
            assert X.shape == (T + 1, 24, 1), (s.name, X.shape)
            assert torch.isfinite(X).all() and torch.isfinite(Y).all(), s.name
        verifie(True, f"axe '{a.name}' : {len(a.scenarios)} scenarios executables")


if __name__ == "__main__":
    CFG.TRAIN_CMD_RANDOMIZE = False   # l'evaluation impose ses commandes via u_seq
    test_non_regression()
    test_formations()
    test_ratio_gps()
    test_corrupt_observations()
    test_rebind()
    test_scenarios()

    print("\n" + "=" * 60)
    if _echecs:
        print(f"{len(_echecs)} ECHEC(S) :")
        for m in _echecs:
            print("  -", m)
        sys.exit(1)
    print("TOUS LES TESTS PASSENT")
