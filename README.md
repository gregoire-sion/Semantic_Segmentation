import torch
import numpy as np
import matplotlib.pyplot as plt


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_drone_errors(test_target, est_out, est_sigma, drone,
                      file_name, sample=0, dt=0.1, n_sigma=3,
                      est_name="EKF"):
    """
    Trace l'erreur (estimee - verite) des 8 composantes d'un drone,
    chacune avec son couloir +/- n_sigma issu de la covariance du filtre.

    test_target : [N, m, T]  etats vrais
    est_out     : [N, m, T]  etats estimes
    est_sigma   : [N, m, m, T]  covariance posterior du filtre
    drone       : 0, 1 ou 2
    sample      : index de la trajectoire a tracer
    """
    comp_names = ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'bx', 'by']
    base = 8 * drone
    cols = [base + k for k in range(8)]

    gt    = _np(test_target[sample])          # [m, T]
    est   = _np(est_out[sample])              # [m, T]
    sig   = _np(est_sigma[sample])            # [m, m, T]
    T = gt.shape[1]
    t = np.arange(T) * dt

    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)
    axes = axes.ravel()

    for k, ci in enumerate(cols):
        ax = axes[k]
        err = est[ci, :] - gt[ci, :]
        std = np.sqrt(np.clip(sig[ci, ci, :], 0, None))   # ecart-type de la composante

        # couloir +/- n_sigma autour de zero (l'erreur ideale est nulle)
        ax.fill_between(t, -n_sigma*std, n_sigma*std,
                        color='#1f77b4', alpha=0.18,
                        label=rf'$\pm{n_sigma}\sigma$')
        ax.axhline(0, color='k', lw=0.8)
        ax.plot(t, err, color='#d62728', lw=1.3, label='erreur')

        # MSE de la composante (dB) dans le titre
        mse = np.mean(err**2)
        mse_db = 10*np.log10(mse) if mse > 0 else -np.inf
        # taux de couverture : fraction de l'erreur dans le couloir
        inside = np.mean(np.abs(err) <= n_sigma*std + 1e-12) * 100
        ax.set_title(f"{comp_names[k]}  |  MSE={mse_db:.1f} dB  |  "
                     f"couv={inside:.0f}%", fontsize=10)
        ax.set_ylabel('err', fontsize=9)
        ax.grid(alpha=0.3)
        if k >= 6:
            ax.set_xlabel('t [s]', fontsize=10)
        if k == 0:
            ax.legend(fontsize=8, loc='upper right')

    fig.suptitle(f"Drone {drone+1}  —  erreur d'estimation {est_name} "
                 f"+/- {n_sigma}sigma (echantillon {sample})",
                 fontsize=14, y=0.997)
    fig.tight_layout()
    fig.savefig(file_name, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return file_name
