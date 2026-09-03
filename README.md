"""
Entraînement de la BASELINE A : archi2, configuration étroite.

Configuration étroite = un seul point de fonctionnement nominal :
  - bruit de mesure fixe (r_scale = 1.0, pas de sweep)
  - une seule famille de commande (les 3 phases historiques)
  - perturbation initiale faible

C'est ce modèle qui sert de référence pour l'étude de généralisation.
Le modèle B (randomisation de domaine) sera entraîné avec le même script
en changeant NARROW -> False.

Usage :
    python train_baseline_narrow.py            # graine 42
    SEED=1234 python train_baseline_narrow.py  # autre graine
"""

import os
import json
import time

import numpy as np
import torch

from KalmanNet_Drones import (
    CFG, SystemModel, EKF, KalmanNetNN,
    generate_dataset, save_dataset, train, run_knet, plot_loss,
)

# --------------------------------------------------------------------------
# Réglages de l'expérience
# --------------------------------------------------------------------------

NARROW = True                 # True = baseline A (étroite) / False = modèle B
ARCHI = "archi2"
SEED = int(os.environ.get("SEED", 42))

# Point de fonctionnement nominal de la config étroite
NARROW_INIT_OFFSET_SCALE = 0.3     # perturbation initiale faible
NARROW_R_SCALE_DB = 0.0            # bruit de mesure nominal

# Config large (modèle B), pour référence
WIDE_NOISE_DB = (-10, 30)
WIDE_INIT_OFFSET_SCALE = 1.0

# Tailles
N_TRAIN = 400
N_VAL = 80
N_TEST = 50                   # >= 50 pour avoir des IC exploitables

RUN_NAME = f"baseline_{'narrow' if NARROW else 'wide'}_{ARCHI}_seed{SEED}"
OUT_DIR = os.path.join("./runs", RUN_NAME)

POS_IDX = [0, 1, 8, 9, 16, 17]


# --------------------------------------------------------------------------
# Application de la config
# --------------------------------------------------------------------------

def apply_config():
    """Écrase les flags de CFG. Tous sont lus à l'exécution sauf OUT_DIR
    et DATASET_PATH, calculés à l'import : on les force explicitement."""
    CFG.SEED = SEED
    CFG.ARCHI_TO_TRAIN = ARCHI
    CFG.N_TRAIN, CFG.N_VAL, CFG.N_TEST = N_TRAIN, N_VAL, N_TEST
    CFG.USE_SAVED_DATASET = False
    CFG.MODE_MONTE_CARLO = False
    CFG.PLOT_MSE_DB = False
    CFG.PLOT_NCI = False

    if NARROW:
        CFG.TRAIN_NOISE_SWEEP = False
        CFG.TRAIN_NOISE_DB = (NARROW_R_SCALE_DB, NARROW_R_SCALE_DB)
        CFG.TRAIN_CMD_RANDOMIZE = False
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


def config_snapshot():
    keys = ["SEED", "N_TRAIN", "N_VAL", "N_TEST", "T", "N_EPOCHS", "N_BATCH",
            "LR", "WD", "GRAD_CLIP", "TBPTT", "IN_MULT", "OUT_MULT",
            "TRAIN_NOISE_SWEEP", "TRAIN_NOISE_DB", "TRAIN_CMD_RANDOMIZE",
            "INIT_OFFSET_P0", "INIT_OFFSET_SCALE"]
    snap = {k: getattr(CFG, k, None) for k in keys}
    snap["TRAIN_CMD_FAMILIES"] = list(getattr(CFG, "TRAIN_CMD_FAMILIES", ()))
    snap["ARCHI"] = ARCHI
    snap["NARROW"] = NARROW
    snap["DEVICE"] = str(CFG.DEVICE)
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in snap.items()}


# --------------------------------------------------------------------------
# Évaluation en distribution
# --------------------------------------------------------------------------

def mse_pos(xhat, xtrue):
    return ((xhat[:, POS_IDX, 0] - xtrue[:, POS_IDX, 0]) ** 2).mean().item()


def evaluate_in_distribution(sm, model, ekf, data_test):
    """MSE position et Delta_dB par trajectoire, sur le test en distribution."""
    Xte, Yte, Ute, Mte = data_test
    n = Xte.shape[0]
    knet, ekf_l, delta = [], [], []
    for i in range(n):
        X, Y, U, M = Xte[i], Yte[i], Ute[i], Mte[i]
        x_ekf, _ = ekf.run(Y, U, M)
        x_knet = run_knet(sm, model, Y, U, M)
        mk, me = mse_pos(x_knet, X), mse_pos(x_ekf, X)
        knet.append(mk); ekf_l.append(me)
        delta.append(10.0 * np.log10(mk / me))
    knet = np.array(knet); ekf_l = np.array(ekf_l); delta = np.array(delta)
    ci = 1.96 * delta.std(ddof=1) / np.sqrt(n)
    return {
        "n_test": n,
        "mse_pos_knet_mean": float(knet.mean()),
        "mse_pos_ekf_mean": float(ekf_l.mean()),
        "delta_db_mean": float(delta.mean()),
        "delta_db_ci95": float(ci),
        "delta_db_per_traj": delta.tolist(),
    }


# --------------------------------------------------------------------------

def main():
    apply_config()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"== Run : {RUN_NAME} ==")
    print(f"   sortie : {OUT_DIR}")
    print(f"   device : {CFG.DEVICE}")
    print(f"   config : noise_sweep={CFG.TRAIN_NOISE_SWEEP} "
          f"cmd_randomize={CFG.TRAIN_CMD_RANDOMIZE} "
          f"init_offset={CFG.INIT_OFFSET_SCALE}")

    sm = SystemModel()
    ekf = EKF(sm)

    print("\n== Génération des données ==")
    t0 = time.time()
    data_train = generate_dataset(sm, CFG.N_TRAIN, seed=SEED,
                                  noise_sweep=CFG.TRAIN_NOISE_SWEEP)
    data_val = generate_dataset(sm, CFG.N_VAL, seed=SEED + 1,
                                noise_sweep=CFG.TRAIN_NOISE_SWEEP)
    # Test en distribution : mêmes réglages que l'entraînement, graine disjointe
    data_test = generate_dataset(sm, CFG.N_TEST, seed=SEED + 99,
                                 noise_sweep=CFG.TRAIN_NOISE_SWEEP)
    save_dataset(data_train, data_val, data_test)
    print(f"   {time.time() - t0:.1f} s")

    print(f"\n== Entraînement {ARCHI} ==")
    model = KalmanNetNN(sm, archi=ARCHI)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"   paramètres : {n_par}")
    t0 = time.time()
    hist_train, hist_val, ckpt = train(sm, model, data_train, data_val, tag=ARCHI)
    train_time = time.time() - t0
    plot_loss(hist_train, hist_val, ARCHI, OUT_DIR)

    state = torch.load(ckpt, map_location=sm.device)
    model.load_state_dict(state["state_dict"])

    print("\n== Évaluation en distribution ==")
    res = evaluate_in_distribution(sm, model, ekf, data_test)
    print(f"   MSE pos KNet : {res['mse_pos_knet_mean']:.4f}")
    print(f"   MSE pos EKF  : {res['mse_pos_ekf_mean']:.4f}")
    print(f"   Delta_dB     : {res['delta_db_mean']:+.2f} "
          f"+/- {res['delta_db_ci95']:.2f} dB (IC95, n={res['n_test']})")
    print("   (Delta_dB < 0 => KalmanNet meilleur que l'EKF)")

    manifest = {
        "run_name": RUN_NAME,
        "checkpoint": ckpt,
        "config": config_snapshot(),
        "n_params": n_par,
        "train_time_s": round(train_time, 1),
        "best_val_loss": float(min(hist_val)),
        "hist_train": hist_train,
        "hist_val": hist_val,
        "in_distribution": res,
    }
    man_path = os.path.join(OUT_DIR, "manifest.json")
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"\n== Manifeste -> {man_path}")
    print(f"== Checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
