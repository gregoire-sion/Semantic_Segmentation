"""
==============================================================================
 GÉNÉRATION DU DATASET PARTAGÉ  ->  dataset.npz
==============================================================================
Crée UNE fois un pool de trajectoires, le découpe en train/val/test SANS remise,
et sauvegarde tout (données + indices du split) dans un fichier .npz unique.

Ce fichier est ensuite chargé À L'IDENTIQUE par :
  - kalmannet_drones.py   (entraînement)
  - analyse_dataset.py    (analyse de diversité)

Garantit que l'analyse et l'entraînement portent sur EXACTEMENT les mêmes
trajectoires et le même découpage.

Usage :  python make_dataset.py
==============================================================================
"""

import os
import numpy as np

from kalmannet_drones import CFG, SystemModel, generate_trajectory

# =============================================================================
# PARAMÈTRES (tout est ici)
# =============================================================================
POOL_SIZE   = 1200          # nb total de trajectoires
FRAC_VAL    = 0.12          # fraction validation
FRAC_TEST   = 0.10          # fraction test
SEED        = 42
DATASET_PATH = os.path.join(CFG.OUT_DIR, "dataset.npz")
NOISE_SWEEP = CFG.TRAIN_NOISE_SWEEP   # cohérent avec l'entraînement


def _np(x):
    if hasattr(x, "a"): return x.a
    if hasattr(x, "cpu"): return x.cpu().numpy()
    return np.asarray(x)


def main():
    os.makedirs(CFG.OUT_DIR, exist_ok=True)
    sm = SystemModel()
    rng = np.random.default_rng(SEED)

    print(f"== Génération du pool : {POOL_SIZE} trajectoires ==")
    if NOISE_SWEEP:
        print(f"   (multi-bruit activé : 1/r² ∈ {CFG.TRAIN_NOISE_DB} dB)")
    lo, hi = CFG.TRAIN_NOISE_DB
    Xs, Ys, Us, Ms = [], [], [], []
    for i in range(POOL_SIZE):
        if NOISE_SWEEP:
            r_scale = 10 ** (-rng.uniform(lo, hi) / 20.0)
        else:
            r_scale = 1.0
        X, Y, U, M = generate_trajectory(sm, rng, r_scale=r_scale)
        Xs.append(_np(X)); Ys.append(_np(Y)); Us.append(_np(U)); Ms.append(_np(M))
        if (i + 1) % 200 == 0:
            print(f"   {i+1}/{POOL_SIZE}")

    X = np.stack(Xs); Y = np.stack(Ys); U = np.stack(Us); M = np.stack(Ms)

    # --- split SANS remise ---
    idx = rng.permutation(POOL_SIZE)
    n_test = int(round(POOL_SIZE * FRAC_TEST))
    n_val  = int(round(POOL_SIZE * FRAC_VAL))
    idx_test  = idx[:n_test]
    idx_val   = idx[n_test:n_test + n_val]
    idx_train = idx[n_test + n_val:]
    assert len(set(idx_train) & set(idx_val)) == 0
    assert len(set(idx_train) & set(idx_test)) == 0
    assert len(set(idx_val) & set(idx_test)) == 0

    print(f"\n== Split ==")
    print(f"   train : {len(idx_train)}  |  val : {len(idx_val)}  |  test : {len(idx_test)}")
    print(f"   recoupements : 0 (garantis par split sans remise)")

    np.savez_compressed(
        DATASET_PATH,
        X=X, Y=Y, U=U, M=M,
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        seed=SEED, noise_sweep=NOISE_SWEEP,
    )
    size_mb = os.path.getsize(DATASET_PATH) / 1e6
    print(f"\n== Sauvegardé -> {DATASET_PATH}  ({size_mb:.1f} Mo) ==")


if __name__ == "__main__":
    main()
