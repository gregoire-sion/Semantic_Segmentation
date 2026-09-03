"""
Entraînement de la BASELINE A : archi2, configuration étroite.

Configuration étroite = un seul point de fonctionnement nominal :
  - bruit de mesure fixe (pas de balayage)
  - une seule famille de commande (les 3 phases historiques)
  - perturbation initiale faible (0.3 x l'écart-type de P0)

Ce modèle sert de référence à toute l'étude de généralisation.
Le modèle B (randomisation de domaine) s'entraîne avec le même script
en mettant NARROW = False.

Lancer trois fois en changeant SEED (42, 1234, 7) pour mesurer la
variabilité due à l'initialisation.
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


# ==========================================================================
# 1. RÉGLAGES DE L'EXPÉRIENCE
#    Les seules lignes à modifier d'un run à l'autre.
# ==========================================================================

SEED = 42                     # changer ici : 42, puis 1234, puis 7
NARROW = True                 # True = baseline A (étroite), False = modèle B
ARCHI = "archi2"

# Tailles des jeux de données
N_TRAIN = 400
N_VAL = 80
N_TEST = 50                   # >= 50 pour avoir un intervalle de confiance utile

# Point de fonctionnement de la configuration ÉTROITE
NARROW_INIT_OFFSET_SCALE = 0.3

# Réglages de la configuration LARGE (modèle B)
WIDE_NOISE_DB = (-10, 30)
WIDE_INIT_OFFSET_SCALE = 1.0

# Où tout est écrit
RUN_NAME = f"baseline_{'narrow' if NARROW else 'wide'}_{ARCHI}_seed{SEED}"
OUT_DIR = os.path.join("./runs", RUN_NAME)


# ==========================================================================
# 2. DÉCOUPAGE DU VECTEUR D'ÉTAT
#    24 composantes = 3 drones x 8 variables (x, y, vx, vy, ax, ay, bx, by).
# ==========================================================================

BASES = (0, 8, 16)            # indice de départ de chaque drone

GROUPES = {
    "position":     [b + i for b in BASES for i in (0, 1)],
    "vitesse":      [b + i for b in BASES for i in (2, 3)],
    "acceleration": [b + i for b in BASES for i in (4, 5)],
    "biais":        [b + i for b in BASES for i in (6, 7)],
}


# ==========================================================================
# 3. APPLICATION DE LA CONFIGURATION
# ==========================================================================

def apply_config():
    """Écrase les réglages de CFG pour ce run.

    Attention : la plupart des attributs de CFG sont lus au moment de
    l'exécution, donc les modifier ici suffit. Mais OUT_DIR et
    DATASET_PATH sont calculés une seule fois, à l'import du module :
    il faut donc les réécrire explicitement, sinon les sorties partent
    dans les anciens dossiers et écrasent d'anciens résultats.
    """
    CFG.SEED = SEED
    CFG.ARCHI_TO_TRAIN = ARCHI
    CFG.N_TRAIN, CFG.N_VAL, CFG.N_TEST = N_TRAIN, N_VAL, N_TEST
    CFG.USE_SAVED_DATASET = False
    CFG.MODE_MONTE_CARLO = False
    CFG.PLOT_MSE_DB = False
    CFG.PLOT_NCI = False

    if NARROW:
        CFG.TRAIN_NOISE_SWEEP = False          # bruit de mesure fixe
        CFG.TRAIN_CMD_RANDOMIZE = False        # commande 3 phases uniquement
        CFG.INIT_OFFSET_P0 = True
        CFG.INIT_OFFSET_SCALE = NARROW_INIT_OFFSET_SCALE
    else:
        CFG.TRAIN_NOISE_SWEEP = True
        CFG.TRAIN_NOISE_DB = WIDE_NOISE_DB
        CFG.TRAIN_CMD_RANDOMIZE = True
        CFG.TRAIN_CMD_FAMILIES = ("phases3_rand", "ou")
        CFG.INIT_OFFSET_P0 = True
        CFG.INIT_OFFSET_SCALE = WIDE_INIT_OFFSET_SCALE

    CFG.OUT_DIR = OUT_DIR
    CFG.DATASET_PATH = os.path.join(OUT_DIR, "dataset.npz")
    os.makedirs(OUT_DIR, exist_ok=True)


# ==========================================================================
# 4. CORRECTION DE LA COURBE DE LOSS D'ENTRAÎNEMENT
# ==========================================================================

def facteur_correction_loss():
    """Nombre de fenêtres TBPTT par séquence.

    Dans train(), la loss d'entraînement est sommée une fois par fenêtre
    TBPTT (8 fenêtres pour T=160 et TBPTT=20) mais divisée seulement par
    le nombre de batches. Elle ressort donc 8 fois trop grande, alors que
    la loss de validation, elle, est bien divisée par T. Sans cette
    correction la figure montre un écart train/val spectaculaire et faux.

    On corrige l'affichage a posteriori plutôt que de modifier train() :
    les poids appris, eux, sont corrects, et les runs déjà effectués
    restent comparables.
    """
    return math.ceil(CFG.T / getattr(CFG, "TBPTT", 20))


# ==========================================================================
# 5. ÉVALUATION EN DISTRIBUTION
# ==========================================================================

def mse_groupe(xhat, xtrue, indices):
    """Erreur quadratique moyenne sur un sous-ensemble de composantes."""
    return ((xhat[:, indices, 0] - xtrue[:, indices, 0]) ** 2).mean().item()


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
    delta_pos = []

    for i in range(n):
        X, Y, U, M = Xte[i], Yte[i], Ute[i], Mte[i]
        x_ekf, _ = ekf.run(Y, U, M)
        x_knet = run_knet(sm, model, Y, U, M)

        for g, idx in GROUPES.items():
            mse_knet[g].append(mse_groupe(x_knet, X, idx))
            mse_ekf[g].append(mse_groupe(x_ekf, X, idx))

        delta_pos.append(10.0 * np.log10(mse_knet["position"][-1]
                                         / mse_ekf["position"][-1]))

    delta_pos = np.array(delta_pos)
    ic95 = 1.96 * delta_pos.std(ddof=1) / np.sqrt(n)

    return {
        "n_test": n,
        "mse_knet": {g: float(np.mean(v)) for g, v in mse_knet.items()},
        "mse_ekf": {g: float(np.mean(v)) for g, v in mse_ekf.items()},
        "delta_db_position": float(delta_pos.mean()),
        "delta_db_ic95": float(ic95),
    }


def afficher_resultats(res):
    print(f"{'groupe':<14} {'MSE KNet':>12} {'MSE EKF':>12} {'gain dB':>10}")
    for g in GROUPES:
        mk, me = res["mse_knet"][g], res["mse_ekf"][g]
        print(f"{g:<14} {mk:>12.4f} {me:>12.4f} {10*np.log10(mk/me):>+10.2f}")
    print()
    print(f"   Delta_dB (position) : {res['delta_db_position']:+.2f} "
          f"+/- {res['delta_db_ic95']:.2f} dB (IC95, n={res['n_test']})")
    print("   Delta_dB < 0  =>  KalmanNet meilleur que l'EKF")


# ==========================================================================
# 6. PROGRAMME PRINCIPAL
# ==========================================================================

def main():
    apply_config()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Les trois jeux utilisent des graines différentes : aucune trajectoire
    # d'entraînement ne peut se retrouver en validation ou en test.
    seed_train, seed_val, seed_test = SEED, SEED + 1, SEED + 99
    assert len({seed_train, seed_val, seed_test}) == 3

    print(f"== Run : {RUN_NAME} ==")
    print(f"   sortie  : {OUT_DIR}")
    print(f"   device  : {CFG.DEVICE}")
    print(f"   config  : noise_sweep={CFG.TRAIN_NOISE_SWEEP} "
          f"cmd_randomize={CFG.TRAIN_CMD_RANDOMIZE} "
          f"init_offset={CFG.INIT_OFFSET_SCALE}")
    print(f"   graines : train={seed_train} val={seed_val} test={seed_test}")

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

    # Correction de l'échelle avant de tracer (voir section 4)
    k = facteur_correction_loss()
    hist_train = [v / k for v in hist_train]
    print(f"   loss train divisée par {k} (fenêtres TBPTT) pour l'affichage")
    fig = plot_loss(hist_train, hist_val, ARCHI, OUT_DIR)

    # On recharge le meilleur checkpoint, pas le modèle de la dernière epoch
    etat = torch.load(ckpt, map_location=sm.device)
    model.load_state_dict(etat["state_dict"])

    # --- Évaluation ------------------------------------------------------
    print("\n== Évaluation en distribution ==")
    res = evaluer(sm, model, ekf, data_test)
    afficher_resultats(res)

    # --- Traçabilité -----------------------------------------------------
    manifeste = {
        "run_name": RUN_NAME,
        "seed": SEED,
        "archi": ARCHI,
        "narrow": NARROW,
        "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
        "T": CFG.T, "n_epochs": CFG.N_EPOCHS, "lr": CFG.LR,
        "noise_sweep": CFG.TRAIN_NOISE_SWEEP,
        "cmd_randomize": CFG.TRAIN_CMD_RANDOMIZE,
        "init_offset_scale": CFG.INIT_OFFSET_SCALE,
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
