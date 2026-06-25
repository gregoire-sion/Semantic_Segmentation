import torch
import numpy as np
import matplotlib.pyplot as plt


def _np(x):
    """Rapatrie un tenseur (GPU ou CPU) en numpy ; laisse passer un array."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_state_component(test_target, kf_out, knet_out, dim,
                         file_name,
                         kf_P=None,        # variance (T,) de la composante 'dim' ; optionnel
                         n_sigma=3,        # largeur du couloir (2 ou 3 typiquement)
                         obs=None,         # observations a superposer ; optionnel
                         sample=0,         # element du batch a afficher
                         dt=1.0,
                         labels=("position", "velocity", "acceleration")):
    """
    Trace une composante de l'etat : trajectoire (haut) + erreur (bas),
    avec couloir +/- n_sigma de l'EKF/KF si kf_P est fourni.
    """
    gt   = _np(test_target[sample][dim, :])
    kf   = _np(kf_out[sample][dim, :])
    knet = _np(knet_out[sample][dim, :])
    T = len(gt)
    t = np.arange(T) * dt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1.4]})

    if kf_P is not None:
        sig = np.sqrt(np.clip(_np(kf_P), 0, None))
        ax1.fill_between(t, kf - n_sigma*sig, kf + n_sigma*sig,
                         color='#1f77b4', alpha=0.18,
                         label=rf'KF $\pm{n_sigma}\sigma$')

    if obs is not None:
        ax1.scatter(t, _np(obs[sample][dim, :]), s=12, c='#cccccc',
                    zorder=1, label='Observations')
    ax1.plot(t, gt,   'k-',  lw=2.4, label='Ground truth', zorder=3)
    ax1.plot(t, kf,   color='#1f77b4', lw=1.7, label='KF', zorder=4)
    ax1.plot(t, knet, color='#2ca02c', lw=1.7, ls='--',
             label='KalmanNet', zorder=5)

    comp = labels[dim] if dim < len(labels) else f'state[{dim}]'
    mse_kf   = np.mean((kf - gt)**2)
    mse_knet = np.mean((knet - gt)**2)
    ax1.set_title(
        f"{comp}  -  MSE  KF={10*np.log10(mse_kf):.1f} dB   "
        f"KNet={10*np.log10(mse_knet):.1f} dB", fontsize=13)
    ax1.set_ylabel(comp, fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=10, ncol=2, loc='best')

    if kf_P is not None:
        ax2.fill_between(t, -n_sigma*sig, n_sigma*sig,
                         color='#1f77b4', alpha=0.18)
    ax2.axhline(0, color='k', lw=0.8)
    ax2.plot(t, kf - gt,   color='#1f77b4', lw=1.3, label='err KF')
    ax2.plot(t, knet - gt, color='#2ca02c', lw=1.3, ls='--', label='err KNet')
    ax2.set_ylabel('erreur', fontsize=12)
    ax2.set_xlabel('t [s]', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig(file_name, dpi=140, bbox_inches='tight')
    plt.close(fig)
