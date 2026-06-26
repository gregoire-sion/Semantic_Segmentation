def plot_drone(drone, X_true, X_ekf, X_knet, P_ekf):
    """Trace les 8 composantes d'un drone donne (0, 1 ou 2)."""
    b = drone * 8          # debut du bloc de ce drone dans l'etat
    d = drone + 1          # numero affiche (1, 2, 3)
    labels = [f"x{d}", f"y{d}", f"vx{d}", f"vy{d}",
              f"ax{d}", f"ay{d}", f"bx{d}", f"by{d}"]

    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    sigma = np.sqrt(P_ekf)

    for k, ax in enumerate(axes.flat):
        idx = b + k
        ax.plot(X_true[idx], "k-", lw=1.5, label="vrai")
        ax.plot(X_ekf[idx], "b--", lw=1, label="EKF")
        ax.fill_between(np.arange(X_ekf.shape[1]),
                        X_ekf[idx] - 3*sigma[idx],
                        X_ekf[idx] + 3*sigma[idx],
                        color="b", alpha=0.15)
        if X_knet is not None:
            ax.plot(X_knet[idx], "r-", lw=1, label="KalmanNet")
        ax.set_title(f"Drone {d} - {labels[k]}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Drone {d}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"comparaison_drone{d}.png", dpi=110)
    plt.show()
