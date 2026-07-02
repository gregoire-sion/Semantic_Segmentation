import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from KalmanNet_Drones import (
    CFG, SystemModel, generate_trajectory, BASES,
)

USE_SAVED    = True
DATASET_PATH = "Dataset/dataset_genere_dans_knet/dataset.npz"

POOL_SIZE = 300
FRAC_TEST = 0.2
SEED = 2025

OUT_DIR = "Dataset/small_dataset/analyse_dataset"


def load_shared(path):
    d = np.load(path)
    train = {"X": d["Xtr"], "U": d["Utr"]}
    test  = {"X": d["Xte"], "U": d["Ute"]}
    print(f">> Dataset chargé : {path}")
    print(f"   train={train['X'].shape[0]}  test={test['X'].shape[0]}")
    return train, test


def build_pool(sm, pool_size, seed):
    rng = np.random.default_rng(seed)
    Xs, Us = [], []
    for _ in range(pool_size):
        X, Y, U, M = generate_trajectory(sm, rng)
        Xs.append(_np(X)); Us.append(_np(U))
    return {
        "X": np.stack(Xs),
        "U": np.stack(Us),
    }


def split_pool(pool, frac_test, seed):
    P = pool["X"].shape[0]
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(P)
    n_test = int(round(P * frac_test))
    idx_test, idx_train = idx[:n_test], idx[n_test:]
    train = {k: v[idx_train] for k, v in pool.items()}
    test = {k: v[idx_test]  for k, v in pool.items()}
    return train, test


def _np(x):
    if hasattr(x, "a"):
        return x.a
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def descriptors(data, sm):
    X, U = data["X"], data["U"]
    N = X.shape[0]
    u = U[..., 0]
    amp_cmd = np.sqrt((u**2).mean(axis=1)).mean(axis=1)
    sign_changes = np.abs(np.diff(np.sign(u), axis=1)).sum(axis=1) / 2
    freq_cmd = sign_changes.mean(axis=1)
    xs = X[..., 0]
    vmean = []
    amean = []
    for b in (0, 8, 16):
        v = np.sqrt(xs[:, :, b+2]**2 + xs[:, :, b+3]**2)
        a = np.sqrt(xs[:, :, b+4]**2 + xs[:, :, b+5]**2)
        vmean.append(v.mean(axis=1)); amean.append(a.mean(axis=1))
    vmean = np.mean(vmean, axis=0)
    amean = np.mean(amean, axis=0)

    def dist(bi, bj):
        return np.sqrt((xs[:, :, bi]-xs[:, :, bj])**2 + (xs[:, :, bi+1]-xs[:, :, bj+1])**2)
    dmean = (dist(0, 8) + dist(8, 16) + dist(0, 16)).mean(axis=1) / 3

    span = []
    for b in (0, 8, 16):
        sx = xs[:, :, b].max(axis=1) - xs[:, :, b].min(axis=1)
        sy = xs[:, :, b+1].max(axis=1) - xs[:, :, b+1].min(axis=1)
        span.append(np.sqrt(sx**2 + sy**2))
    span = np.mean(span, axis=0)

    return {
        "amp_cmd": amp_cmd, "freq_cmd": freq_cmd,
        "vmean": vmean, "amean": amean, "dmean": dmean, "span": span,
    }


def fig1_commandes(dtr, dte, outdir):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, titre, xlab in [
        (axs[0], "amp_cmd", "Amplitude des commandes (RMS)", "amplitude"),
        (axs[1], "freq_cmd", "Fréquence des commandes (passages à zéro)", "≈ fréquence"),
    ]:
        ax.hist(dtr[key], bins=25, alpha=0.6, color='steelblue', label='train', density=True)
        ax.hist(dte[key], bins=25, alpha=0.6, color='crimson',   label='test',  density=True)
        ax.set_title(titre); ax.set_xlabel(xlab); ax.set_ylabel("densité")
        ax.grid(True, ls=':', alpha=.6); ax.legend()
    fig.suptitle("Diversité des commandes u — train et test", fontweight='bold')
    fig.tight_layout()
    p = os.path.join(outdir, "1_commandes.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig2_trajectoires(train, test, sm, outdir, n_show=40):
    fig, ax = plt.subplots(figsize=(9, 8))

    Xtr = train["X"][..., 0]; Xte = test["X"][..., 0]
    for i in range(min(n_show, Xtr.shape[0])):
        for b, c in [(0, 'steelblue'), (8, 'lightgreen'), (16, 'plum')]:
            ax.plot(Xtr[i, :, b], Xtr[i, :, b+1], color=c, alpha=0.12, lw=0.8)

    for i in range(Xte.shape[0]):
        for b, c in [(0, 'navy'), (8, 'darkgreen'), (16, 'purple')]:
            lbl = None
            ax.plot(Xte[i, :, b], Xte[i, :, b+1], color=c, alpha=0.5, lw=1.0,
                    label=lbl)

    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color='steelblue', alpha=.8, label='train (clair)'),
               Line2D([0],[0], color='navy', label='test (foncé)')]
    ax.legend(handles=handles)
    ax.set_title("Trajectoires spatiales x-y — train (clair) et test (foncé)",
                 fontweight='bold')
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.grid(True, ls=':', alpha=.6)
    ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    p = os.path.join(outdir, "2_trajectoires.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig3_etat(dtr, dte, outdir):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, key, titre, xlab in [
        (axs[0], "vmean", "Vitesse moyenne", "|v| (m/s)"),
        (axs[1], "amean", "Accélération moyenne", "|a| (m/s²)"),
        (axs[2], "dmean", "Distance inter-drones moyenne", "d (m)"),
    ]:
        ax.hist(dtr[key], bins=25, alpha=0.6, color='steelblue', label='train', density=True)
        ax.hist(dte[key], bins=25, alpha=0.6, color='crimson',   label='test',  density=True)
        ax.set_title(titre); ax.set_xlabel(xlab); ax.set_ylabel("densité")
        ax.grid(True, ls=':', alpha=.6); ax.legend()
    fig.suptitle("Couverture de l'espace d'état — train et test", fontweight='bold')
    fig.tight_layout()
    p = os.path.join(outdir, "3_espace_etat.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig4_descripteurs(dtr, dte, outdir):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    pairs = [("amp_cmd", "freq_cmd", "amplitude cmd", "fréquence cmd"),
             ("vmean", "dmean", "vitesse moy.", "distance inter-drones moy.")]
    for ax, (kx, ky, lx, ly) in zip(axs, pairs):
        ax.scatter(dtr[kx], dtr[ky], s=18, c='steelblue', alpha=.5, label='train')
        ax.scatter(dte[kx], dte[ky], s=28, c='crimson', alpha=.8,
                   edgecolors='k', linewidths=.4, label='test')
        ax.set_xlabel(lx); ax.set_ylabel(ly)
        ax.grid(True, ls=':', alpha=.6); ax.legend()
    fig.suptitle("Descripteurs — train et test", fontweight='bold')
    fig.tight_layout()
    p = os.path.join(outdir, "4_descripteurs.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig5_span(dtr, dte, outdir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(dtr["span"], bins=25, alpha=0.6, color='steelblue', label='train', density=True)
    ax.hist(dte["span"], bins=25, alpha=0.6, color='crimson',   label='test',  density=True)
    ax.set_title("Étendue spatiale des trajectoires")
    ax.set_xlabel("span (m)"); ax.set_ylabel("densité")
    ax.legend(); ax.grid(True, ls=':', alpha=.6)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "5_span.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sm = SystemModel()

    if USE_SAVED:
        train, test = load_shared(DATASET_PATH)
    else:
        print(f"== Génération d'un pool de {POOL_SIZE} trajectoires ==")
        pool = build_pool(sm, POOL_SIZE, SEED)
        train, test = split_pool(pool, FRAC_TEST, SEED)
    print(f"   train : {train['X'].shape[0]} traj  |  test : {test['X'].shape[0]} traj")

    dtr = descriptors(train, sm)
    dte = descriptors(test, sm)

    produced = [
        fig1_commandes(dtr, dte, OUT_DIR),
        fig2_trajectoires(train, test, sm, OUT_DIR),
        fig3_etat(dtr, dte, OUT_DIR),
        fig4_descripteurs(dtr, dte, OUT_DIR),
        fig5_span(dtr, dte, OUT_DIR),
    ]

    print("\n== Figures ==")
    for p in produced:
        print("  ", p)


if __name__ == "__main__":
    main()

