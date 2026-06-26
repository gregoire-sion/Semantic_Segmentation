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
    for drone in range(SM.N_DRONES):
        plot_drone(drone, X_true, X_ekf, X_knet, P_ekf, Y_np)
    plt.show()
    print("\nFigure sauvegardee : comparaison_drone2.png")


def plot_drone(drone, X_true, X_ekf, X_knet, P_ekf, Y):
    """Trace les 8 composantes d'un drone donne (0, 1 ou 2)."""
    b = drone * 8          # debut du bloc de ce drone dans l'etat
    d = drone + 1          # numero affiche (1, 2, 3)
    labels = [f"x{d}", f"y{d}", f"vx{d}", f"vy{d}",
              f"ax{d}", f"ay{d}", f"bx{d}", f"by{d}"]

    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    sigma = np.sqrt(P_ekf)
    T=X_ekf.shape[1]
    tgrid = np.arange(T)

    for k, ax in enumerate(axes.flat):
        idx = b + k
        #ax.plot(X_true[idx], "k-", lw=1.5, label="vrai")
        ax.plot(X_ekf[idx] - X_true[idx], "g--", lw=1, label="EKF")
        ax.fill_between(np.arange(X_ekf.shape[1]),
                        - 3*sigma[idx],
                        3*sigma[idx],
                        color="b", alpha=0.15, label=r'Couloir $\pm 3\sigma$')
        if X_knet is not None:
            ax.plot(X_knet[idx], "r-", lw=1, label="KalmanNet")
        if drone == 0 and k==0:
            ax.scatter(tgrid, Y[0] - X_true[0], s=8, c="red", alpha=0.4, label="mesure GPS")
        if drone == 0 and k==1:
            ax.scatter(tgrid, Y[1] - X_true[1], s=8, c="red", alpha=0.4, label="mesure GPS")
        if drone == 1 and k==4:
            ax.scatter(tgrid, Y[2] - X_true[12], s=8, c="red", alpha=0.4, label="mesure IMU (biaisée)")
        if drone == 1 and k==5:
            ax.scatter(tgrid, Y[3] - X_true[13], s=8, c="red", alpha=0.4, label="mesure IMU (biaisée)")
        ax.set_title(f"Drone {d} - {labels[k]}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Drone {d}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"comparaison_drone{d}.png", dpi=110)
    


if __name__ == "__main__":
    main()

    """
Etat        : m = 24  (3 drones x 8 : x, y, vx, vy, ax, ay, bx, by)
Observation : n = 8   (GPS drone1, accelero biaise drone2 (x,y),
                       distances d12, d23, d13, d23_redondant)

Format batche PyTorch : tenseurs [batch, m, 1] / [batch, n, 1].
"""

import torch
import numpy as np

# =========================================================================
# DIMENSIONS
# =========================================================================
M = 24 
N = 8 
N_DRONES = 3
DT = 0.1 

DIST_EPS = 1e-2

_COMMAND_SEQUENCE = None


def build_command_sequence(T, dt=DT):

    t = torch.arange(T, dtype=torch.float32) * dt
    u = torch.zeros(T, M)

    A1, w1 = 1.0, 0.5
    u[:, 4] = A1 * torch.sin(w1 * t)
    u[:, 5] = A1 * torch.cos(w1 * t)

    A3, w3 = 1.0, 0.7
    u[:, 20] = A3 * torch.sin(w3 * t)
    u[:, 21] = A3 * torch.cos(w3 * t)

    return u


def set_command_sequence(u_seq):
    """Enregistre la sequence courante (partagee data-gen / filtre)."""
    global _COMMAND_SEQUENCE
    _COMMAND_SEQUENCE = u_seq


def reset_command():
    """Remet le pointeur de commande a zero (debut de trajectoire)."""
    global _COMMAND_SEQUENCE
    _COMMAND_SEQUENCE = None


def get_command(t):
    """Renvoie la commande u_t [M, 1] a l'instant t, ou 0 si non definie."""
    if _COMMAND_SEQUENCE is None:
        return torch.zeros(M, 1)
    return _COMMAND_SEQUENCE[t].reshape(M, 1)


# =========================================================================
# DYNAMIQUE : f(x, t)  ->  etat suivant
# =========================================================================
def f(x, t=0):
    """Evolution d'etat (batchee). x : [batch, M, 1] -> [batch, M, 1].

    Replaye la commande deterministe a l'instant t (drones 1 et 3).
    C'est le point critique : f DOIT utiliser get_command(t), pas un
    u_current fige.
    """
    batch = x.shape[0]
    x_next = x.clone()

    u_t = get_command(t).to(x.device).unsqueeze(0)   # [1, M, 1] -> broadcast

    for d in [0,2] :
        b = d * 8
        # Indices locaux : px,py,vx,vy,ax,ay,bx,by = b+0..b+7
        px, py = x[:, b+0], x[:, b+1]
        vx, vy = x[:, b+2], x[:, b+3]
        ax, ay = x[:, b+4], x[:, b+5]
        bx, by = x[:, b+6], x[:, b+7]
        # bx, by (biais) : random walk -> inchanges en moyenne

        # Integration cinematique (Euler) -> [A RAFFINER si tu utilises
        # un schema d'ordre superieur dans ta simu]
        x_next[:, b+0] = px + DT * vx + 0.5 * DT**2 * ax
        x_next[:, b+1] = py + DT * vy + 0.5 * DT**2 * ay
        x_next[:, b+2] = vx + DT * ax
        x_next[:, b+3] = vy + DT * ay
        x_next[:, b+4] = 0
        x_next[:, b+5] = 0
        x_next[:, b+6] = bx
        x_next[:, b+7] = by
        # ax,ay,bx,by : random walk -> propages tels quels (le bruit Q
        # est ajoute par le data-gen, pas ici)

    for d in [1] :
        b = d * 8
        # Indices locaux : px,py,vx,vy,ax,ay,bx,by = b+0..b+7
        px, py = x[:, b+0], x[:, b+1]
        vx, vy = x[:, b+2], x[:, b+3]
        ax, ay = x[:, b+4], x[:, b+5]
        bx, by = x[:, b+6], x[:, b+7]
        # bx, by (biais) : random walk -> inchanges en moyenne

        # Integration cinematique (Euler) -> [A RAFFINER si tu utilises
        # un schema d'ordre superieur dans ta simu]
        x_next[:, b+0] = px + DT * vx + 0.5 * DT**2 * ax
        x_next[:, b+1] = py + DT * vy + 0.5 * DT**2 * ay
        x_next[:, b+2] = vx + DT * ax
        x_next[:, b+3] = vy + DT * ay
        x_next[:, b+4] = ax
        x_next[:, b+5] = ay
        x_next[:, b+6] = bx
        x_next[:, b+7] = by
        # ax,ay,bx,by : random walk -> propages tels quels (le bruit Q
        # est ajoute par le data-gen, pas ici)

    # Ajout des commandes deterministes (drones 1 et 3)
    x_next = x_next + u_t  # u_t agit sur ax/ay des drones 1 et 3

    return x_next


def jacobian_f(x, t=0):
    """Jacobien de f evalue en x. Retourne [batch, M, M].

    [À COMPLÉTER] : porter ici ton Jacobien analytique deja valide en
    NumPy (compare aux differences finies). Squelette fourni avec la
    structure cinematique bloc-diagonale.
    """
    batch = x.shape[0]
    F = torch.zeros(batch, M, M, device=x.device)

    for d in range(N_DRONES):
        b = d * 8
        # Bloc cinematique 8x8 par drone
        # px depend de px, vx, ax
        F[:, b+0, b+0] = 1.0
        F[:, b+0, b+2] = DT
        F[:, b+0, b+4] = 0.5 * DT**2
        # py depend de py, vy, ay
        F[:, b+1, b+1] = 1.0
        F[:, b+1, b+3] = DT
        F[:, b+1, b+5] = 0.5 * DT**2
        # vx depend de vx, ax
        F[:, b+2, b+2] = 1.0
        F[:, b+2, b+4] = DT
        # vy depend de vy, ay
        F[:, b+3, b+3] = 1.0
        F[:, b+3, b+5] = DT
        # ax, ay
        F[:, b+4, b+4] = 0.0
        F[:, b+5, b+5] = 0.0
        # bx, by 
        F[:, b+6, b+6] = 1.0
        F[:, b+7, b+7] = 1.0
        if d==1 :
            F[:, b+4, b+4] = 1.0
            F[:, b+5, b+5] = 1.0

    return F


# =========================================================================
# OBSERVATION : h(x)  ->  mesure
# =========================================================================
def _dist(x, da, db):
    """Distance euclidienne entre drone da et drone db (batchee, stable)."""
    bxa, bxb = da * 8, db * 8
    dx = x[:, bxa+0] - x[:, bxb+0]
    dy = x[:, bxa+1] - x[:, bxb+1]
    # sqrt(.. + eps) : evite gradient infini quand les drones coincident
    return torch.sqrt(dx**2 + dy**2 + DIST_EPS**2)


def h(x):
    """Modele d'observation (batche). x : [batch, M, 1] -> [batch, N, 1].

    n = 8 :
      0,1 : GPS position drone1 (x1, y1)
      2,3 : accelero biaise drone2 (ax2+bx2, ay2+by2)
      4   : distance d12
      5   : distance d23
      6   : distance d13
      7   : distance d23 redondante (mesuree par drone3)
    """
    batch = x.shape[0]
    y = torch.zeros(batch, N, 1, device=x.device)

    # GPS drone1 (indices 0,1)
    y[:, 0, 0] = x[:, 0, 0]
    y[:, 1, 0] = x[:, 1, 0]

    # Accelero biaise drone2 : a + biais (indices ax2=12+4=16, bx2=12+6=18)
    y[:, 2, 0] = x[:, 16, 0] + x[:, 18, 0]   # ax2 + bx2
    y[:, 3, 0] = x[:, 17, 0] + x[:, 19, 0]   # ay2 + by2

    # Distances inter-drones
    y[:, 4, 0] = _dist(x, 0, 1)[:, 0]        # d12
    y[:, 5, 0] = _dist(x, 1, 2)[:, 0]        # d23
    y[:, 6, 0] = _dist(x, 0, 2)[:, 0]        # d13
    y[:, 7, 0] = _dist(x, 1, 2)[:, 0]        # d23 redondant

    return y


def jacobian_h(x):
    """Jacobien de h evalue en x. Retourne [batch, N, M].

    [À COMPLÉTER] : porter ton Jacobien analytique NumPy. Les lignes GPS
    et accelero sont triviales (lineaires) ; les lignes distances sont
    non lineaires et fournies ci-dessous.
    """
    batch = x.shape[0]
    H = torch.zeros(batch, N, M, device=x.device)

    # GPS drone1 (lineaire)
    H[:, 0, 0] = 1.0
    H[:, 1, 1] = 1.0

    # Accelero drone2 (lineaire)
    H[:, 2, 12] = 1.0   # d(ax2+bx2)/d ax2
    H[:, 2, 14] = 1.0   # d(ax2+bx2)/d bx2
    H[:, 3, 13] = 1.0
    H[:, 3, 15] = 1.0

    # --- Lignes distances (non lineaires) ---
    def fill_dist_row(row, da, db):
        bxa, bxb = da * 8, db * 8
        dx = x[:, bxa+0, 0] - x[:, bxb+0, 0]
        dy = x[:, bxa+1, 0] - x[:, bxb+1, 0]
        dist = torch.sqrt(dx**2 + dy**2 + DIST_EPS**2)
        H[:, row, bxa+0] = dx / dist
        H[:, row, bxa+1] = dy / dist
        H[:, row, bxb+0] = -dx / dist
        H[:, row, bxb+1] = -dy / dist

    fill_dist_row(4, 0, 1)   # d12
    fill_dist_row(5, 1, 2)   # d23
    fill_dist_row(6, 0, 2)   # d13
    fill_dist_row(7, 1, 2)   # d23 redondant

    return H


def get_Q(q2=1e-3):
    Q = torch.eye(M) * q2
    for b in (0, 8, 16):
        Q[b+4, b+4] = Q[b+5, b+5] = 0.5**2
        Q[b+6, b+6] = Q[b+7, b+7] = 1e-5**2
    Q[10, 10] = Q[11, 11] = 0.1**2
    Q[12, 12] = Q[13, 13] = 1e-2**2
    return Q


def get_R(r2=1e-2):
    R = torch.eye(N)
    sigma_R_gps = 0.5
    sigma_R_acc = 0.1
    sigma_R_d   = 0.5
    R[0,0] = sigma_R_gps**2
    R[1,1] = sigma_R_gps**2
    R[2,2] = sigma_R_acc**2
    R[3,3] = sigma_R_acc**2
    R[4,4] = sigma_R_d**2
    R[5,5] = sigma_R_d**2
    R[6,6] = sigma_R_d**2
    R[7,7] = sigma_R_d**2
    return R

def get_x0_true(batch=1):
    x0 = torch.zeros(batch, M, 1)
    #drone1
    x0[:,0,0] = 0
    x0[:,1,0] = 10
    x0[:,2,0] = 0
    x0[:,3,0] = 0
    x0[:,4,0] = 0
    x0[:,5,0] = 0
    x0[:,6,0] = 0
    x0[:,7,0] = 0
    #drone2
    x0[:,8,0] = 10
    x0[:,9,0] = 0
    x0[:,10,0] = 1
    x0[:,11,0] = 0
    x0[:,12,0] = 0
    x0[:,13,0] = 0
    x0[:,14,0] = 0.5
    x0[:,15,0] = -0.2
    #drone3
    x0[:,16,0] = 0
    x0[:,17,0] = -10
    x0[:,18,0] = 1
    x0[:,19,0] = 0
    x0[:,20,0] = 0
    x0[:,21,0] = 0
    x0[:,22,0] = 0
    x0[:,23,0] = 0

    return x0
# =========================================================================
# GENERATION DE DONNEES
# =========================================================================
def generate(T, batch=1, x0=None, q2=1e-3, r2=1e-2, seed=None):
    """Genere (X, Y) : trajectoires etat + observations bruitees.

    X : [batch, M, T+1]   (etats ground-truth, x0 inclus)
    Y : [batch, N, T]     (observations bruitees)

    Replaye la sequence de commandes via set_command_sequence -> garantit
    que la generation et le filtrage utilisent la MEME commande.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Sequence de commandes partagee
    u_seq = build_command_sequence(T)
    set_command_sequence(u_seq)

    if x0 is None:
        x0 = get_x0_true(batch)        

    X = torch.zeros(batch, M, T + 1)
    Y = torch.zeros(batch, N, T)
    X[:, :, 0] = x0[:, :, 0]

    Q = get_Q(q2)

    Lq = torch.linalg.cholesky(get_Q(q2))
    Lr = torch.linalg.cholesky(get_R(r2))

    x = x0
    for t in range(T):
        # Bruit d'etat
        w = (Lq @ torch.randn(batch, M, 1))
        x = f(x, t) + w
        X[:, :, t + 1] = x[:, :, 0]
        # Bruit d'observation
        v = (Lr @ torch.randn(batch, N, 1))
        y = h(x) + v
        Y[:, :, t] = y[:, :, 0]

    reset_command()
    return X, Y


if __name__ == "__main__":
    # Test rapide de coherence (analogue a ta validation NumPy)
    print(f"Dimensions : M={M}, N={N}, N_DRONES={N_DRONES}")
    X, Y = generate(T=50, batch=2, seed=0)
    print(f"X shape : {tuple(X.shape)}  (attendu [2, {M}, 51])")
    print(f"Y shape : {tuple(Y.shape)}  (attendu [2, {N}, 50])")
    print(f"X fini : {torch.isfinite(X).all().item()}")
    print(f"Y fini : {torch.isfinite(Y).all().item()}")

    # Verif Jacobien f par differences finies (1 echantillon)
    x_test = torch.randn(1, M, 1)
    Fa = jacobian_f(x_test, t=0)[0]
    eps = 1e-5
    Fn = torch.zeros(M, M)
    f0 = f(x_test, t=0)[0, :, 0]
    for j in range(M):
        xp = x_test.clone(); xp[0, j, 0] += eps
        Fn[:, j] = (f(xp, t=0)[0, :, 0] - f0) / eps
    err = (Fa - Fn).abs().max().item()
    print(f"Erreur max Jacobien f (analytique vs diff. finies) : {err:.2e}")
