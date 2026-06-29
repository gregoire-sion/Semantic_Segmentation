"""
==============================================================================
 COMPARAISON archi1 vs archi2  (+ EKF baseline)
==============================================================================
Recharge les deux modèles KalmanNet entraînés (knet_archi1.pt / knet_archi2.pt)
produits par kalmannet_drones.py, génère une trajectoire de test commune, et
trace des comparaisons côte à côte :
  - erreur par drone (archi1 vs archi2 vs EKF) avec couloir +-3sigma EKF
  - MSE de position par drone (barres)
  - (optionnel) couloirs Monte-Carlo empiriques des deux archis

Lance d'abord kalmannet_drones.py (ARCHI_TO_TRAIN="both") pour produire les .pt.
==============================================================================
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# on réutilise tout le moteur du fichier principal
from kalmannet_drones import (
    CFG, SystemModel, EKF, KalmanNetNN,
    generate_trajectory, run_knet, monte_carlo_knet,
    LABELS, BASES,
)


def load_model(sm, archi, outdir=CFG.OUT_DIR):
    ckpt = os.path.join(outdir, f"knet_{archi}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"{ckpt} introuvable. Lance d'abord kalmannet_drones.py "
            f"avec ARCHI_TO_TRAIN='both' (ou '{archi}').")
    model = KalmanNetNN(sm, archi=archi)
    state = torch.load(ckpt, map_location=sm.device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model


def plot_compare_drone(sm, drone, X, x_ekf, x_a1, x_a2, P_ekf, temps,
                       sig_mc1=None, sig_mc2=None, outdir=CFG.OUT_DIR):
    """4x2 sous-plots, erreur des deux archis + EKF + couloir 3sigma EKF."""
    base = BASES[drone]
    fig, axs = plt.subplots(4, 2, figsize=(13, 9), sharex=True)
    axs = axs.flatten()
    for i in range(8):
        idx = base + i
        e_ekf = (x_ekf[:, idx, 0] - X[:, idx, 0]).cpu().numpy()
        e_a1  = (x_a1[:, idx, 0]  - X[:, idx, 0]).cpu().numpy()
        e_a2  = (x_a2[:, idx, 0]  - X[:, idx, 0]).cpu().numpy()
        sig   = np.sqrt(P_ekf[:, idx, idx].cpu().numpy())
        axs[i].fill_between(temps, -3*sig, 3*sig, color='blue', alpha=0.12,
                            label=r'$\pm 3\sigma$ EKF')
        if sig_mc1 is not None:
            axs[i].plot(temps,  3*sig_mc1[:, idx], color='orange', ls='--', lw=.8)
            axs[i].plot(temps, -3*sig_mc1[:, idx], color='orange', ls='--', lw=.8)
        if sig_mc2 is not None:
            axs[i].plot(temps,  3*sig_mc2[:, idx], color='red', ls=':', lw=.8)
            axs[i].plot(temps, -3*sig_mc2[:, idx], color='red', ls=':', lw=.8)
        axs[i].plot(temps, e_ekf, color='green',  lw=1.2, label='EKF')
        axs[i].plot(temps, e_a1,  color='orange', lw=1.2, label='KNet archi1')
        axs[i].plot(temps, e_a2,  color='red',    lw=1.2, label='KNet archi2')
        axs[i].axhline(0, color='k', lw=.6)
        axs[i].set_title(f"{LABELS[i]} : estimé − vrai", fontsize=10)
        axs[i].grid(True, ls=':', alpha=.7)
    axs[6].set_xlabel("Temps (s)"); axs[7].set_xlabel("Temps (s)")
    fig.suptitle(f"Drone {drone} — comparaison archi1 / archi2 / EKF",
                 fontsize=13, fontweight='bold')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, .965))
    fig.tight_layout(rect=[0, 0, 1, .93])
    p = os.path.join(outdir, f"compare_drone{drone}.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def plot_mse_bars(mse_dict, outdir=CFG.OUT_DIR):
    """Barres : MSE position par drone pour EKF / archi1 / archi2."""
    drones = [1, 2, 3]
    methods = list(mse_dict.keys())
    x = np.arange(len(drones)); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, mth in enumerate(methods):
        vals = [mse_dict[mth][d] for d in drones]
        ax.bar(x + (j-1)*w, vals, w, label=mth)
    ax.set_xticks(x); ax.set_xticklabels([f"Drone {d}" for d in drones])
    ax.set_ylabel("MSE position"); ax.set_yscale('log')
    ax.set_title("MSE de position par drone — EKF vs KalmanNet")
    ax.grid(True, ls=':', alpha=.7, axis='y'); ax.legend()
    fig.tight_layout()
    p = os.path.join(outdir, "compare_mse_bars.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main():
    os.makedirs(CFG.OUT_DIR, exist_ok=True)
    torch.manual_seed(CFG.SEED); np.random.seed(CFG.SEED)
    sm = SystemModel()

    # trajectoire de test commune
    rng = np.random.default_rng(CFG.SEED + 99)
    Xte, Yte, Ute, Mte = generate_trajectory(sm, rng)
    temps = np.arange(CFG.T + 1) * sm.dt

    # EKF
    ekf = EKF(sm)
    x_ekf, P_ekf = ekf.run(Yte, Ute, Mte)

    # modèles
    m1 = load_model(sm, "archi1")
    m2 = load_model(sm, "archi2")
    x_a1 = run_knet(sm, m1, Yte, Ute, Mte)
    x_a2 = run_knet(sm, m2, Yte, Ute, Mte)

    # Monte-Carlo togglable
    sig1 = sig2 = None
    if CFG.MODE_MONTE_CARLO:
        print(f"Monte-Carlo {CFG.N_MC} runs par archi...")
        sig1 = monte_carlo_knet(sm, m1)
        sig2 = monte_carlo_knet(sm, m2)

    produced = []
    mse_dict = {"EKF": {}, "archi1": {}, "archi2": {}}
    for d in (1, 2, 3):
        produced.append(plot_compare_drone(sm, d, Xte, x_ekf, x_a1, x_a2,
                                            P_ekf, temps, sig1, sig2))
        b = BASES[d]
        mse_dict["EKF"][d]    = ((x_ekf[:, b:b+2, 0]-Xte[:, b:b+2, 0])**2).mean().item()
        mse_dict["archi1"][d] = ((x_a1[:, b:b+2, 0]-Xte[:, b:b+2, 0])**2).mean().item()
        mse_dict["archi2"][d] = ((x_a2[:, b:b+2, 0]-Xte[:, b:b+2, 0])**2).mean().item()

    produced.append(plot_mse_bars(mse_dict))

    print("\n=== MSE position (log) ===")
    for d in (1, 2, 3):
        print(f"  Drone {d} : EKF={mse_dict['EKF'][d]:.4f} | "
              f"archi1={mse_dict['archi1'][d]:.4f} | "
              f"archi2={mse_dict['archi2'][d]:.4f}")
    print("\n== Figures ==")
    for p in produced:
        print("  ", p)


if __name__ == "__main__":
    main()
