"""
ENTRAÎNEMENT — TROIS VARIANTES

Un seul script, trois configurations d'entraînement possibles, choisies
par la constante VARIANTE en haut du fichier :

  "narrow"      baseline étroite : un seul point de fonctionnement
  "wide"        randomisation du bruit et de la commande, offset fixé à 1,0
  "offsetsweep" comme "wide", mais l'offset initial est lui aussi TIRÉ
                au hasard dans [0, 2] à chaque trajectoire

POURQUOI LA TROISIÈME VARIANTE
Le balayage a montré que la baseline décroche dès un offset de 0,73, et
que le modèle "wide" ne fait guère mieux (0,90) — alors qu'il a été
entraîné à un offset de 1,0. Mais "wide" ne RANDOMISE pas l'offset : il
le déplace seulement de 0,3 à 1,0. Le test honnête n'a donc jamais été
fait.

La variante "offsetsweep" tranche entre deux hypothèses :
  - problème de représentativité → montrer de grandes erreurs initiales
    suffit, la frontière doit reculer nettement au-delà de 2
  - problème structurel → les features sont normalisées L2, le réseau
    reste aveugle à l'ampleur de son erreur, et la frontière ne bougera
    pas beaucoup

Les deux issues sont un résultat exploitable.

PRÉREQUIS
Cette variante suppose que KalmanNet_Drones.py a été modifié pour
accepter TRAIN_OFFSET_SWEEP et TRAIN_OFFSET_RANGE (voir
patch_kalmannet_drones.md). Sans la modification, le script s'arrête avec
un message explicite plutôt que d'entraîner silencieusement un modèle à
offset fixe.

À LANCER (trois fois, en changeant SEED) :
    python entrainement.py
"""

import os
import json
import math
import time

import numpy as np
import torch

from KalmanNet_Drones import (
    CFG, SystemModel, EKF, KalmanNetNN,
    generate_dataset, save_dataset, train, run_knet, plot_loss,
)
from metriques import (
    GROUPES, mse_par_groupe, delta_db, moyenne_ic95, afficher_tableau,
)


# ==========================================================================
# 1. RÉGLAGES DE L'EXPÉRIENCE
#    Les deux seules lignes à modifier d'un run à l'autre.
# ==========================================================================

SEED = 42                     # 42, puis 1234, puis 7
VARIANTE = "narrow"           # "narrow" | "wide" | "offsetsweep"

ARCHI = "archi2"

N_TRAIN = 400
N_VAL = 80
N_TEST = 50                   # >= 50 pour un intervalle de confiance utile

RUN_NAME = f"baseline_{VARIANTE}_{ARCHI}_seed{SEED}"
OUT_DIR = os.path.join("./runs", RUN_NAME)


# ==========================================================================
# 2. LES TROIS CONFIGURATIONS
# ==========================================================================
# Chaque entrée décrit entièrement une variante. Mettre les trois côte à
# côte dans un seul dictionnaire rend les différences lisibles d'un coup
# d'œil, au lieu de les disséminer dans des branches if/else.
#
# offset_range : None  -> offset fixe, valeur donnée par offset_scale
#                (a, b) -> offset tiré au hasard dans [a, b] par trajectoire

CONFIGURATIONS = {
    "narrow": {
        "bruit_sweep": False,
        "bruit_db": (0.0, 0.0),
        "commande_randomisee": False,
        "familles": (),
        "offset_scale": 0.3,
        "offset_range": None,
    },
    "wide": {
        "bruit_sweep": True,
        "bruit_db": (-10.0, 30.0),
        "commande_randomisee": True,
        "familles": ("phases3_rand", "ou"),
        "offset_scale": 1.0,
        "offset_range": None,
    },
    "offsetsweep": {
        "bruit_sweep": True,
        "bruit_db": (-10.0, 30.0),
        "commande_randomisee": True,
        "familles": ("phases3_rand", "ou"),
        "offset_scale": 1.0,        # ignoré quand offset_range est défini
        "offset_range": (0.0, 2.0),
    },
}


def appliquer_configuration():
    """Écrase les réglages de CFG pour la variante choisie.

    Attention : la plupart des attributs de CFG sont lus au moment de
    l'exécution, donc les modifier ici suffit. Mais OUT_DIR et
    DATASET_PATH sont calculés une seule fois, à l'import du module :
    il faut donc les réécrire explicitement, sinon les sorties partent
    dans les anciens dossiers et écrasent d'anciens résultats.
    """
    if VARIANTE not in CONFIGURATIONS:
        raise SystemExit(f"Variante inconnue : {VARIANTE!r}. "
                         f"Choisir parmi {sorted(CONFIGURATIONS)}.")
    config = CONFIGURATIONS[VARIANTE]

    CFG.SEED = SEED
    CFG.ARCHI_TO_TRAIN = ARCHI
    CFG.N_TRAIN, CFG.N_VAL, CFG.N_TEST = N_TRAIN, N_VAL, N_TEST
    CFG.USE_SAVED_DATASET = False
    CFG.MODE_MONTE_CARLO = False
    CFG.PLOT_MSE_DB = False
    CFG.PLOT_NCI = False

    CFG.TRAIN_NOISE_SWEEP = config["bruit_sweep"]
    CFG.TRAIN_NOISE_DB = config["bruit_db"]
    CFG.TRAIN_CMD_RANDOMIZE = config["commande_randomisee"]
    if config["familles"]:
        CFG.TRAIN_CMD_FAMILIES = config["familles"]

    CFG.INIT_OFFSET_P0 = True
    CFG.INIT_OFFSET_SCALE = config["offset_scale"]

    if config["offset_range"] is None:
        CFG.TRAIN_OFFSET_SWEEP = False
    else:
        verifier_patch_applique()
        CFG.TRAIN_OFFSET_SWEEP = True
        CFG.TRAIN_OFFSET_RANGE = config["offset_range"]

    CFG.OUT_DIR = OUT_DIR
    CFG.DATASET_PATH = os.path.join(OUT_DIR, "dataset.npz")
    os.makedirs(OUT_DIR, exist_ok=True)
    return config


def verifier_patch_applique():
    """Vérifie que generate_trajectory sait tirer l'offset au hasard.

    Sans cette vérification, un oubli du patch produirait silencieusement
    un modèle à offset fixe portant le nom "offsetsweep" — une erreur
    coûteuse à détecter, puisqu'elle ne se verrait qu'aux résultats.
    """
    import inspect
    from KalmanNet_Drones import generate_trajectory
    code = inspect.getsource(generate_trajectory)
    if "TRAIN_OFFSET_SWEEP" not in code:
        raise SystemExit(
            "generate_trajectory ne gère pas TRAIN_OFFSET_SWEEP.\n"
            "Applique d'abord la modification décrite dans "
            "patch_kalmannet_drones.md, sinon l'offset resterait fixe.")


def resume_configuration(config):
    """Description lisible de la configuration, pour l'affichage et le JSON."""
    if config["offset_range"] is None:
        offset = f"fixe à {config['offset_scale']}"
    else:
        lo, hi = config["offset_range"]
        offset = f"tiré dans [{lo}, {hi}]"

    bruit = (f"tiré dans {config['bruit_db']} dB"
             if config["bruit_sweep"] else "fixe à 0 dB")
    commande = (f"tirée parmi {list(config['familles'])}"
                if config["commande_randomisee"] else "3 phases nominale")
    return {"bruit": bruit, "commande": commande, "offset": offset}


# ==========================================================================
# 3. CORRECTION DE LA COURBE DE LOSS
# ==========================================================================

def facteur_correction_loss():
    """Nombre de fenêtres TBPTT par séquence.

    Dans train(), la loss d'entraînement est sommée une fois par fenêtre
    TBPTT (8 fenêtres pour T=160 et TBPTT=20) mais divisée seulement par
    le nombre de batches. Elle ressort donc 8 fois trop grande, alors que
    la loss de validation est bien divisée par T. Sans cette correction,
    la figure montre un écart train/val spectaculaire et faux.

    On corrige l'affichage a posteriori plutôt que de modifier train() :
    les poids appris sont corrects, et les runs déjà faits restent
    comparables.
    """
    return math.ceil(CFG.T / getattr(CFG, "TBPTT", 20))


# ==========================================================================
# 4. ÉVALUATION EN DISTRIBUTION
# ==========================================================================

def evaluer(sm, model, ekf, data_test):
    """Compare KalmanNet et l'EKF sur chaque trajectoire de test.

    Le Delta_dB est calculé trajectoire par trajectoire, puis moyenné.
    C'est ce qui permet d'assortir le résultat d'un intervalle de
    confiance : une moyenne sans dispersion ne se compare à rien.
    """
    Xte, Yte, Ute, Mte = data_test
    n = Xte.shape[0]

    mse_knet = {g: [] for g in GROUPES}
    mse_ekf = {g: [] for g in GROUPES}
    deltas = []

    for i in range(n):
        X, Y, U, M = Xte[i], Yte[i], Ute[i], Mte[i]
        x_ekf, _ = ekf.run(Y, U, M)
        x_knet = run_knet(sm, model, Y, U, M)

        mk, me = mse_par_groupe(x_knet, X), mse_par_groupe(x_ekf, X)
        for g in GROUPES:
            mse_knet[g].append(mk[g])
            mse_ekf[g].append(me[g])
        deltas.append(delta_db(mk["position"], me["position"]))

    moyenne, ic95 = moyenne_ic95(deltas)

    return {
        "n_test": n,
        "mse_knet": {g: float(np.mean(v)) for g, v in mse_knet.items()},
        "mse_ekf": {g: float(np.mean(v)) for g, v in mse_ekf.items()},
        "delta_db_position": moyenne,
        "delta_db_ic95": ic95,
    }


# ==========================================================================
# 5. PROGRAMME PRINCIPAL
# ==========================================================================

def main():
    config = appliquer_configuration()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Graines distinctes : aucune trajectoire d'entraînement ne peut se
    # retrouver en validation ou en test.
    seed_train, seed_val, seed_test = SEED, SEED + 1, SEED + 99
    assert len({seed_train, seed_val, seed_test}) == 3

    description = resume_configuration(config)
    print(f"== Run : {RUN_NAME} ==")
    print(f"   sortie   : {OUT_DIR}")
    print(f"   device   : {CFG.DEVICE}")
    print(f"   bruit    : {description['bruit']}")
    print(f"   commande : {description['commande']}")
    print(f"   offset   : {description['offset']}")
    print(f"   graines  : train={seed_train} val={seed_val} test={seed_test}")

    sm = SystemModel()
    ekf = EKF(sm)

    # --- Données ---------------------------------------------------------
    print("\n== Génération des données ==")
    t0 = time.time()
    data_train = generate_dataset(sm, N_TRAIN, seed=seed_train,
                                  noise_sweep=CFG.TRAIN_NOISE_SWEEP)
    data_val = generate_dataset(sm, N_VAL, seed=seed_val,
                                noise_sweep=CFG.TRAIN_NOISE_SWEEP)
    data_test = generate_dataset(sm, N_TEST, seed=seed_test,
                                 noise_sweep=CFG.TRAIN_NOISE_SWEEP)
    save_dataset(data_train, data_val, data_test)
    print(f"   {time.time() - t0:.1f} s")

    # --- Entraînement ----------------------------------------------------
    print(f"\n== Entraînement {ARCHI} ==")
    model = KalmanNetNN(sm, archi=ARCHI)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   paramètres : {n_params}")

    t0 = time.time()
    hist_train, hist_val, ckpt = train(sm, model, data_train, data_val, tag=ARCHI)
    duree = time.time() - t0

    k = facteur_correction_loss()
    hist_train = [v / k for v in hist_train]
    print(f"   loss train divisée par {k} (fenêtres TBPTT) pour l'affichage")
    fig = plot_loss(hist_train, hist_val, ARCHI, OUT_DIR)

    # On recharge le meilleur checkpoint, pas le modèle de la dernière epoch.
    etat = torch.load(ckpt, map_location=sm.device)
    model.load_state_dict(etat["state_dict"])

    # --- Évaluation ------------------------------------------------------
    # Attention : le jeu de test suit la MÊME configuration que
    # l'entraînement. Pour la variante offsetsweep, il contient donc des
    # offsets variés — ce Delta_dB n'est pas directement comparable à
    # celui des deux autres variantes. La comparaison équitable se fait
    # via les balayages, où les conditions de test sont identiques.
    print("\n== Évaluation en distribution ==")
    res = evaluer(sm, model, ekf, data_test)
    afficher_tableau(res["mse_knet"], res["mse_ekf"])
    print()
    print(f"   Delta_dB (position) : {res['delta_db_position']:+.2f} "
          f"+/- {res['delta_db_ic95']:.2f} dB (IC95, n={res['n_test']})")
    print("   Delta_dB < 0  =>  KalmanNet meilleur que l'EKF")

    # --- Traçabilité -----------------------------------------------------
    manifeste = {
        "run_name": RUN_NAME,
        "variante": VARIANTE,
        "seed": SEED,
        "archi": ARCHI,
        "configuration": config,
        "description": description,
        "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
        "T": CFG.T, "n_epochs": CFG.N_EPOCHS, "lr": CFG.LR,
        "n_params": n_params,
        "train_time_s": round(duree, 1),
        "best_val_loss": float(min(hist_val)),
        "hist_train": hist_train,
        "hist_val": hist_val,
        "resultats": res,
        "checkpoint": ckpt,
    }
    chemin = os.path.join(OUT_DIR, "manifest.json")
    with open(chemin, "w", encoding="utf-8") as fh:
        json.dump(manifeste, fh, indent=2, ensure_ascii=False)

    print(f"\n== Manifeste  -> {chemin}")
    print(f"== Figure     -> {fig}")
    print(f"== Checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
