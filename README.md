"""
SystemModel.py
==============
Source UNIQUE de verite pour le probleme a 3 drones.

Tout (KalmanNet.py, Train.py, Simulation.py) importe f/h/Jacobiens d'ICI.
Objectif : garantir que l'EKF baseline et KalmanNet partagent EXACTEMENT
la meme dynamique et la meme sequence de commandes -> evite la resurgence
du bug de desynchronisation command-sequence.

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
M = 24          # dimension etat
N = 8           # dimension observation
N_DRONES = 3
DT = 0.1        # pas de temps [s]  -> [A AJUSTER selon ta simu]

# Epsilon distances : releve a 1e-2 (au lieu de 1e-6) pour stabiliser
# le Jacobien de h quand deux drones se rapprochent. Suspect NaN #1.
DIST_EPS = 1e-2


# =========================================================================
# SEQUENCE DE COMMANDES (deterministe, drones 1 et 3 uniquement)
# =========================================================================
# Drones 1 et 3 ont des commandes d'acceleration sinusoidales ZERO-MEAN
# (obligatoire : une acceleration moyenne non nulle fait diverger la position).
# Drone 2 : PAS de commande (acceleration = random walk inconnu a estimer).

_COMMAND_SEQUENCE = None   # rempli par set_command_sequence


def build_command_sequence(T, dt=DT):
    """Construit la sequence de commandes [T, m] pour toute la trajectoire.

    Retourne un tenseur [T, M] ou seules les composantes (ax, ay) des
    drones 1 et 3 sont non nulles. A appeler une fois, puis injecter via
    set_command_sequence() dans le data-gen ET dans le filtre.
    """
    t = torch.arange(T, dtype=torch.float32) * dt
    u = torch.zeros(T, M)

    # --- Drone 1 : indices ax=4, ay=5 ---
    # [A COMPLETER] amplitudes / pulsations selon ta simu validee NumPy
    A1, w1 = 1.0, 0.5
    u[:, 4] = A1 * torch.sin(w1 * t)        # ax1
    u[:, 5] = A1 * torch.cos(w1 * t)        # ay1

    # --- Drone 3 : indices ax=20, ay=21 ---
    A3, w3 = 1.0, 0.7
    u[:, 20] = A3 * torch.sin(w3 * t)       # ax3
    u[:, 21] = A3 * torch.cos(w3 * t)       # ay3

    # Drone 2 (indices 12..19) : aucune commande -> reste a zero.
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

    for d in range(N_DRONES):
        b = d * 8
        # Indices locaux : px,py,vx,vy,ax,ay,bx,by = b+0..b+7
        px, py = x[:, b+0], x[:, b+1]
        vx, vy = x[:, b+2], x[:, b+3]
        ax, ay = x[:, b+4], x[:, b+5]
        # bx, by (biais) : random walk -> inchanges en moyenne

        # Integration cinematique (Euler) -> [A RAFFINER si tu utilises
        # un schema d'ordre superieur dans ta simu]
        x_next[:, b+0] = px + DT * vx + 0.5 * DT**2 * ax
        x_next[:, b+1] = py + DT * vy + 0.5 * DT**2 * ay
        x_next[:, b+2] = vx + DT * ax
        x_next[:, b+3] = vy + DT * ay
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
        # ax, ay, bx, by : random walk -> identite
        F[:, b+4, b+4] = 1.0
        F[:, b+5, b+5] = 1.0
        F[:, b+6, b+6] = 1.0
        F[:, b+7, b+7] = 1.0

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
    H[:, 2, 16] = 1.0   # d(ax2+bx2)/d ax2
    H[:, 2, 18] = 1.0   # d(ax2+bx2)/d bx2
    H[:, 3, 17] = 1.0
    H[:, 3, 19] = 1.0

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


# =========================================================================
# COVARIANCES INITIALES (pour l'EKF baseline ; KalmanNet ne s'en sert pas)
# =========================================================================
def get_Q(q2=1e-3):
    """Bruit d'etat Q = q2 * I  [M, M]. [A AJUSTER]."""
    return q2 * torch.eye(M)


def get_R(r2=1e-2):
    """Bruit d'observation R = r2 * I  [N, N]. [A AJUSTER]."""
    return r2 * torch.eye(N)


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
        x0 = torch.zeros(batch, M, 1)
        # [A COMPLETER] positions/vitesses initiales realistes par drone

    X = torch.zeros(batch, M, T + 1)
    Y = torch.zeros(batch, N, T)
    X[:, :, 0] = x0[:, :, 0]

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
