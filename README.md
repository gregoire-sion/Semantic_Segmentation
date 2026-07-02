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

from kalmannet_drones import (CFG, SystemModel, generate_trajectory,
                              build_command_ood)

# =============================================================================
# PARAMÈTRES (tout est ici)
# =============================================================================
POOL_SIZE   = 1200          # nb total de trajectoires
FRAC_VAL    = 0.12          # fraction validation
FRAC_TEST   = 0.10          # fraction test
N_OOD       = 5             # trajectoires OOD par type (3phases, brutal)
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

    # --- Bloc OOD (régimes dynamiques non vus, HORS train/val/test) --------
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
    print(f"\n== Sauvegardé -> {DATASET_PATH}  ({size_mb:.1f} Mo) ==")


if __name__ == "__main__":
    main()

"""
==============================================================================
 TEST OUT-OF-DISTRIBUTION (OOD)
==============================================================================
Évalue KalmanNet (et l'EKF) sur des régimes dynamiques JAMAIS vus à
l'entraînement : commandes 3-phases et manœuvres brutales.

Ces trajectoires ne comptent PAS dans les métriques de test officielles —
elles sont un résultat en soi, une sonde de robustesse hors distribution.

Produit :
  - plots par drone (erreur EKF vs KNet) pour chaque type de OOD
  - chiffre de DÉGRADATION : de combien la MSE empire vs le test in-distribution

Prérequis :
  - dataset.npz généré par make_dataset.py (contient le bloc OOD)
  - modèles entraînés knet_archi1.pt / knet_archi2.pt
==============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from kalmannet_drones import (
    CFG, SystemModel, EKF, KalmanNetNN, run_knet,
    plot_drone, BASES, LABELS,
)

# =============================================================================
# PARAMÈTRES
# =============================================================================
DATASET_PATH = os.path.join(CFG.OUT_DIR, "dataset.npz")
ARCHI        = "archi1"        # archi du modèle à tester en OOD
OOD_KINDS    = ["3phases", "brutal"]
OUT_DIR      = os.path.join(CFG.OUT_DIR, "test_ood")


def load_model(sm, archi):
    ckpt = os.path.join(CFG.OUT_DIR, f"knet_{archi}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"{ckpt} introuvable. Entraîne d'abord le modèle.")
    model = KalmanNetNN(sm, archi=archi)
    state = torch.load(ckpt, map_location=sm.device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model


def to_t(a, sm):
    return torch.tensor(a, dtype=torch.float32, device=sm.device)


def mse_position(x_est, X):
    """MSE de position moyennée sur les 3 drones."""
    idx = [0, 1, 8, 9, 16, 17]
    return ((x_est[:, idx, 0] - X[:, idx, 0])**2).mean().item()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sm = SystemModel()
    d = np.load(DATASET_PATH)
    temps = np.arange(CFG.T + 1) * sm.dt

    ekf = EKF(sm)
    model = load_model(sm, ARCHI)

    # --- référence in-distribution : MSE sur le test normal ----------------
    ite = d["idx_test"]
    mse_id_knet, mse_id_ekf = [], []
    for j in ite:
        X = to_t(d["X"][j], sm); Y = to_t(d["Y"][j], sm)
        U = to_t(d["U"][j], sm); M = to_t(d["M"][j], sm)
        xe, _ = ekf.run(Y, U, M)
        xk = run_knet(sm, model, Y, U, M)
        mse_id_ekf.append(mse_position(xe, X))
        mse_id_knet.append(mse_position(xk, X))
    mse_id_knet = np.mean(mse_id_knet)
    mse_id_ekf  = np.mean(mse_id_ekf)

    print("=" * 60)
    print(f"Référence IN-DISTRIBUTION (test normal) :")
    print(f"  MSE pos : KNet={mse_id_knet:.3f}  EKF={mse_id_ekf:.3f}")
    print("=" * 60)

    produced = []
    summary = []

    # --- pour chaque type de OOD ------------------------------------------
    for kind in OOD_KINDS:
        Xo = d[f"Xood_{kind}"]; Yo = d[f"Yood_{kind}"]
        Uo = d[f"Uood_{kind}"]; Mo = d[f"Mood_{kind}"]
        n = Xo.shape[0]

        mse_knet, mse_ekf = [], []
        # on garde la 1re trajectoire pour les plots détaillés
        for i in range(n):
            X = to_t(Xo[i], sm); Y = to_t(Yo[i], sm)
            U = to_t(Uo[i], sm); M = to_t(Mo[i], sm)
            xe, Pe = ekf.run(Y, U, M)
            xk = run_knet(sm, model, Y, U, M)
            mse_ekf.append(mse_position(xe, X))
            mse_knet.append(mse_position(xk, X))
            if i == 0:
                for drone in (1, 2, 3):
                    p = plot_drone(sm, drone, X, xe, xk, Pe, temps,
                                   f"OOD_{kind}", outdir=OUT_DIR)
                    produced.append(p)

        mk, me = np.mean(mse_knet), np.mean(mse_ekf)
        # dégradation vs in-distribution (facteur multiplicatif)
        deg_knet = mk / mse_id_knet
        deg_ekf  = me / mse_id_ekf
        summary.append((kind, mk, me, deg_knet, deg_ekf))

        print(f"\nOOD '{kind}' ({n} trajectoires) :")
        print(f"  MSE pos KNet={mk:.3f}  (x{deg_knet:.1f} vs in-distrib)")
        print(f"  MSE pos EKF ={me:.3f}  (x{deg_ekf:.1f} vs in-distrib)")

    # --- figure de synthèse : dégradation ---------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    kinds = [s[0] for s in summary]
    x = np.arange(len(kinds)); w = 0.35
    ax.bar(x - w/2, [s[3] for s in summary], w, label='KalmanNet', color='crimson')
    ax.bar(x + w/2, [s[4] for s in summary], w, label='EKF', color='green')
    ax.axhline(1.0, color='k', ls='--', lw=1, label='niveau in-distribution')
    ax.set_xticks(x); ax.set_xticklabels(kinds)
    ax.set_ylabel("Facteur de dégradation MSE (× vs in-distrib)")
    ax.set_title(f"Dégradation hors distribution — {ARCHI}")
    ax.legend(); ax.grid(True, ls=':', alpha=.6, axis='y')
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "degradation_ood.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    produced.append(p)

    print("\n== Figures ==")
    for p in produced:
        print("  ", p)


if __name__ == "__main__":
    main()

