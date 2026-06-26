"""
Simulation.py
=============
Pipeline d'EVALUATION FINALE (jeu de TEST uniquement).

Compare sur EXACTEMENT la meme trajectoire de test :
  - EKF classique (baseline model-based)
  - KalmanNet entraine (poids charges depuis knet_best.pt)

Produit : MSE [dB] global et par composante, couverture ±3sigma (EKF),
et figures comparatives.

L'EKF et KalmanNet tirent leur f/h de SystemModel -> comparaison honnete,
meme dynamique, meme sequence de commandes.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import SystemModel as SM
from KalmanNet import KalmanNet


# =========================================================================
# EKF BASELINE (model-based, mono-trajectoire pour lisibilite)
# =========================================================================
def run_ekf(Y, x0, P0, q2=1e-3, r2=1e-2):
    """EKF classique sur une trajectoire. Y : [N, T].

    Retourne X_hat [M, T] et la covariance diagonale stockee P_diag [M, T]
    (pour tracer les corridors ±3sigma).
    """
    T = Y.shape[1]
    Q = SM.get_Q(q2).numpy()
    R = SM.get_R(r2).numpy()

    x = x0.copy().reshape(SM.M, 1)
    P = P0.copy()

    X_hat = np.zeros((SM.M, T))
    P_diag = np.zeros((SM.M, T))

    for t in range(T):
        # --- Prediction ---
        xt = torch.tensor(x.reshape(1, SM.M, 1), dtype=torch.float32)
        x_prior = SM.f(xt, t).numpy().reshape(SM.M, 1)
        F = SM.jacobian_f(xt, t).numpy()[0]
        P_prior = F @ P @ F.T + Q

        # --- Update ---
        xp = torch.tensor(x_prior.reshape(1, SM.M, 1), dtype=torch.float32)
        y_prior = SM.h(xp).numpy().reshape(SM.N, 1)
        H = SM.jacobian_h(xp).numpy()[0]

        S = H @ P_prior @ H.T + R
        K = P_prior @ H.T @ np.linalg.inv(S)

        innov = Y[:, t].reshape(SM.N, 1) - y_prior
        x = x_prior + K @ innov
        P = (np.eye(SM.M) - K @ H) @ P_prior

        X_hat[:, t] = x[:, 0]
        P_diag[:, t] = np.diag(P)

    return X_hat, P_diag


# =========================================================================
# KALMANNET (charge les poids entraines)
# =========================================================================
def run_knet(Y, weights_path, gru_mult=2):
    """Execute KalmanNet entraine sur une trajectoire. Y : [N, T] -> [M, T]."""
    net = KalmanNet(gru_mult=gru_mult)
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    net.eval()

    SM.set_command_sequence(SM.build_command_sequence(Y.shape[1]))
    with torch.no_grad():
        Yb = torch.tensor(Y.reshape(1, SM.N, -1), dtype=torch.float32)
        X_hat = net(Yb).numpy()[0]
    SM.reset_command()
    return X_hat


# =========================================================================
# METRIQUES
# =========================================================================
def mse_db(X_hat, X_true):
    """MSE global en dB."""
    err = X_hat - X_true
    return 10.0 * np.log10(np.mean(err**2))


def coverage_3sigma(X_hat, P_diag, X_true):
    """Pourcentage de points dans le corridor ±3sigma (par composante)."""
    sigma = np.sqrt(P_diag)
    inside = np.abs(X_hat - X_true) <= 3.0 * sigma
    return 100.0 * inside.mean(axis=1)   # [M]


# =========================================================================
# PIPELINE DE TEST
# =========================================================================
def main(weights_path="knet_best.pt", T_test=277, gru_mult=2,
         q2=1e-3, r2=1e-2, seed=123):
    print("=== Generation de la trajectoire de TEST ===")
    X, Y = SM.generate(T=T_test, batch=1, q2=q2, r2=r2, seed=seed)
    X_true = X[0, :, 1:].numpy()    # [M, T] : etats apres f
    Y_np = Y[0].numpy()             # [N, T]

    x0 = np.zeros(SM.M)
    P0 = np.eye(SM.M) * 1.0

    print("=== EKF baseline ===")
    X_ekf, P_ekf = run_ekf(Y_np, x0, P0, q2=q2, r2=r2)
    db_ekf = mse_db(X_ekf, X_true)
    cov_ekf = coverage_3sigma(X_ekf, P_ekf, X_true)
    print(f"MSE EKF       : {db_ekf:.2f} dB")
    print(f"Couverture ±3σ EKF (moyenne) : {cov_ekf.mean():.1f} %")

    # KalmanNet : seulement si les poids existent
    import os
    if os.path.exists(weights_path):
        print("=== KalmanNet ===")
        X_knet = run_knet(Y_np, weights_path, gru_mult=gru_mult)
        db_knet = mse_db(X_knet, X_true)
        print(f"MSE KalmanNet : {db_knet:.2f} dB")
        print(f"\nGain KalmanNet vs EKF : {db_ekf - db_knet:+.2f} dB")
    else:
        print(f"(Poids '{weights_path}' introuvables -> EKF seul. "
              f"Lance Train.py d'abord.)")
        X_knet = None

    # --- Figure : focus drone 2 (le drone a biais inconnu, ton objectif) ---
    plot_drone2(X_true, X_ekf, X_knet, P_ekf)
    print("\nFigure sauvegardee : comparaison_drone2.png")


def plot_drone2(X_true, X_ekf, X_knet, P_ekf):
    """Trace les composantes cle du drone 2 (indices 12..19)."""
    b = 12   # debut drone 2
    labels = ["x2", "y2", "vx2", "vy2", "ax2", "ay2", "bx2", "by2"]
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
        ax.set_title(f"Drone 2 - {labels[k]}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("comparaison_drone2.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
