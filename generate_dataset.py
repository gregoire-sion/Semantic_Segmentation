import os
import numpy as np

from KalmanNet_Drones import (CFG, SystemModel, generate_trajectory,
                              build_command_ood)

POOL_SIZE = 150
FRAC_VAL = 0.12
FRAC_TEST = 0.10
N_OOD = 5
SEED  = 42
DATASET_PATH = CFG.DATASET_PATH
NOISE_SWEEP = CFG.TRAIN_NOISE_SWEEP


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

    print(f"\n== Génération OOD : {N_OOD} traj/type (jamais dans les métriques) ==")
    ood = {}
    for kind in ("3phases", "brutal"):
        Xo, Yo, Uo, Mo = [], [], [], []
        for _ in range(N_OOD):
            u = build_command_ood(CFG.T, sm.dt, rng, kind=kind)
            Xg, Yg, Ug, Mg = generate_trajectory(sm, rng, u_seq=u)
            Xo.append(_np(Xg)); Yo.append(_np(Yg)); Uo.append(_np(Ug)); Mo.append(_np(Mg))
        ood[f"Xood_{kind}"] = np.stack(Xo)
        ood[f"Yood_{kind}"] = np.stack(Yo)
        ood[f"Uood_{kind}"] = np.stack(Uo)
        ood[f"Mood_{kind}"] = np.stack(Mo)
        print(f"   {kind} : {N_OOD} trajectoires")

    np.savez_compressed(
        DATASET_PATH,
        X=X, Y=Y, U=U, M=M,
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        seed=SEED, noise_sweep=NOISE_SWEEP,
        **ood,
    )
    size_mb = os.path.getsize(DATASET_PATH) / 1e6
    print(f"\n== Sauvegardé -> {DATASET_PATH}")


if __name__ == "__main__":
    main()