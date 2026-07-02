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
    X, U = d["X"], d["U"]
    itr, ite = d["idx_train"], d["idx_test"]
    train = {"X": X[itr], "U": U[itr]}
    test  = {"X": X[ite], "U": U[ite]}
    print(f">> Dataset partagé chargé : {path}")
    print(f"   train={len(itr)}  test={len(ite)}  (split identique à l'entraînement)")
    return train, test, itr, ite

def build_pool(sm, pool_size, seed):

    rng = np.random.default_rng(seed)
    Xs, Us = [], []
    for _ in range(pool_size):
        X, Y, U, M = generate_trajectory(sm, rng)
        Xs.append(_np(X)); Us.append(_np(U))
    return {
        "X": np.stack(Xs),      # [P, T+1, m, 1]
        "U": np.stack(Us),      # [P, T, 6, 1]
    }


def split_pool(pool, frac_test, seed):
    P = pool["X"].shape[0]
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(P)
    n_test = int(round(P * frac_test))
    idx_test, idx_train = idx[:n_test], idx[n_test:]
    train = {k: v[idx_train] for k, v in pool.items()}
    test = {k: v[idx_test]  for k, v in pool.items()}
    return train, test, idx_train, idx_test


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
    fig.suptitle("Diversité des commandes u — train vs test", fontweight='bold')
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
    ax.set_title("Trajectoires spatiales x-y — train (clair) vs test (foncé)",
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
    fig.suptitle("Couverture de l'espace d'état — train vs test", fontweight='bold')
    fig.tight_layout()
    p = os.path.join(outdir, "3_espace_etat.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig4_chevauchement(dtr, dte, outdir):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    pairs = [("amp_cmd", "freq_cmd", "amplitude cmd", "fréquence cmd"),
             ("vmean", "dmean", "vitesse moy.", "distance inter-drones moy.")]
    for ax, (kx, ky, lx, ly) in zip(axs, pairs):
        ax.scatter(dtr[kx], dtr[ky], s=18, c='steelblue', alpha=.5, label='train')
        ax.scatter(dte[kx], dte[ky], s=28, c='crimson', alpha=.8,
                   edgecolors='k', linewidths=.4, label='test')
        ax.set_xlabel(lx); ax.set_ylabel(ly)
        ax.grid(True, ls=':', alpha=.6); ax.legend()
    fig.suptitle("Chevauchement train / test dans l'espace des descripteurs",
                 fontweight='bold')
    fig.tight_layout()
    p = os.path.join(outdir, "4_chevauchement.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p

def coverage_metrics(dtr, dte):

    keys = ["amp_cmd", "freq_cmd", "vmean", "amean", "dmean", "span"]

    Xtr = np.stack([dtr[k] for k in keys], axis=1)
    Xte = np.stack([dte[k] for k in keys], axis=1)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd

    lo, hi = Xtr.min(0), Xtr.max(0)
    inside = np.all((Xte >= lo) & (Xte <= hi), axis=1)
    pct_inside = 100 * inside.mean()

    dmin = []
    for z in Zte:
        d = np.sqrt(((Ztr - z)**2).sum(axis=1))
        dmin.append(d.min())
    dmin = np.array(dmin)
    return pct_inside, dmin, keys

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sm = SystemModel()

    if USE_SAVED:
        train, test, itr, ite = load_shared(DATASET_PATH)
    else:
        print(f"== Génération d'un pool de {POOL_SIZE} trajectoires ==")
        pool = build_pool(sm, POOL_SIZE, SEED)
        train, test, itr, ite = split_pool(pool, FRAC_TEST, SEED)
    print(f"   train : {train['X'].shape[0]} traj  |  test : {test['X'].shape[0]} traj")
    print(f"   intersection d'indices : {len(set(itr.tolist()) & set(ite.tolist()))}  (doit être 0)")

    dtr = descriptors(train, sm)
    dte = descriptors(test, sm)

    produced = [
        fig1_commandes(dtr, dte, OUT_DIR),
        fig2_trajectoires(train, test, sm, OUT_DIR),
        fig3_etat(dtr, dte, OUT_DIR),
        fig4_chevauchement(dtr, dte, OUT_DIR),
    ]

    pct_inside, dmin, keys = coverage_metrics(dtr, dte)
    print("\n=== MÉTRIQUES DE SÉPARATION / COUVERTURE ===")
    print(f"  % de trajectoires test dans l'enveloppe du train : {pct_inside:.1f}%")
    print(f"  distance min test->train (normalisée) :")
    print(f"     min={dmin.min():.3f}  médiane={np.median(dmin):.3f}  (proche de 0 = quasi-doublon)")
    if dmin.min() < 0.05:
        print("      des trajectoires test sont très proches du train (quasi-doublons)")
    else:
        print("      aucune trajectoire test n'est un quasi-doublon du train")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(dmin, bins=20, color='teal', alpha=.7)
    ax.axvline(0.05, color='r', ls='--', label='seuil quasi-doublon')
    ax.set_title("Distance minimale test → train (espace descripteurs normalisé)")
    ax.set_xlabel("distance min"); ax.set_ylabel("nb de trajectoires test")
    ax.legend(); ax.grid(True, ls=':', alpha=.6)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "5_separation.png"); fig.savefig(p, dpi=130); plt.close(fig)
    produced.append(p)

    print("\n== Figures ==")
    for p in produced:
        print("  ", p)


if __name__ == "__main__":
    main()
