"""
==============================================================================
 KalmanNet — Localisation collaborative 3 drones (système COMMANDÉ par u)
==============================================================================

Adaptation de l'architecture KalmanNet (Revach et al., 2022) au modèle d'état
centralisé à 3 drones de Grégoire. Différences clés avec le repo KalmanNet_TSP :

  1.  Le système est COMMANDÉ : f(x, u) = F.x + B.u  (le repo original n'a que f(x)).
  2.  Cadence asynchrone des capteurs gérée par MASQUAGE de l'innovation :
        - IMU (drone 2)         -> chaque pas
        - GPS (drone 1) + dist  -> tous les 5 pas
      L'observation y garde TOUJOURS la dimension n=7 ; aux pas sans GPS/dist on
      met l'innovation correspondante à 0 (masque). Le réseau apprend à mettre
      ~0 de gain sur les canaux masqués.
  3.  Deux architectures RNN sélectionnables :
        - ARCHI 1 : un seul GRU, features {F2, F4}
        - ARCHI 2 : trois GRU en cascade Q -> Sigma -> S, toutes les features
  4.  EKF baseline (mêmes f,h,Q,R) pour fournir le couloir +-3 sigma de référence.
  5.  Mode Monte-Carlo TOGGLABLE (le code tourne identiquement sans l'activer).

État (m=24) : 3 drones x [x, y, vx, vy, ax, ay, bx, by]
Obs  (n=7)  : [GPSx1, GPSy1, IMUx2(=ax2+bx2), IMUy2(=ay2+by2), d12, d23, d13]

Format tenseurs : [batch, dim, 1] partout (convention de Grégoire).
==============================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

# =============================================================================
# 0. CONFIGURATION GLOBALE  (tout ce que tu touches au quotidien est ici)
# =============================================================================
class CFG:
    # --- Choix d'architecture(s) à entraîner -------------------------------
    # "archi1", "archi2", ou "both"
    ARCHI_TO_TRAIN = "both"

    # --- Monte-Carlo (couloir empirique au TEST) ---------------------------
    MODE_MONTE_CARLO = False     # <-- togglable : True/False, le code tourne dans les 2 cas
    N_MC             = 30        # nb de runs MC (ignoré si MODE_MONTE_CARLO=False)

    # --- Métriques additionnelles inspirées du papier (togglables) ---------
    PLOT_MSE_DB   = True   # courbe MSE[dB] vs 1/r^2 (Fig.5-9 du papier) : KNet rejoint la MMSE de l'EKF
    R_SWEEP_DB    = [-10, 0, 10, 20, 30]   # niveaux 1/r^2 [dB] balayés pour la courbe MSE[dB]
    N_MC_DB       = 15     # runs MC par niveau de bruit pour la courbe MSE[dB]
    PLOT_NCI      = True   # cohérence statistique : NCI / taux de couverture 3sigma (nécessite MC)

    # --- Données -----------------------------------------------------------
    N_TRAIN   = 200      # trajectoires d'entraînement (commandes u variées)
    N_VAL     = 30
    N_TEST    = 1        # 1 trajectoire de référence pour les plots détaillés
    T         = 160      # longueur d'une trajectoire (n_steps)
    SEED      = 42

    # --- Entraînement ------------------------------------------------------
    N_EPOCHS   = 60
    N_BATCH    = 20
    LR         = 1e-3
    WD         = 1e-4    # weight decay (régularisation L2, eq.(13) du papier)
    GRAD_CLIP  = 5.0     # clip du gradient (stabilise la BPTT, cf. NaN connus)

    # --- Multiplicateurs de largeur du réseau (cf. args du repo) ----------
    IN_MULT   = 5
    OUT_MULT  = 40

    # --- Sorties -----------------------------------------------------------
    OUT_DIR   = "/mnt/user-data/outputs"
    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. PARAMÈTRES PHYSIQUES DU MODÈLE  (repris de ton EKF centralisé)
# =============================================================================
class SystemModel:
    """Porte f, h, F, B, Q, R, et toute la dynamique. Tout en torch."""

    def __init__(self, device=CFG.DEVICE):
        self.device = device
        self.dt        = 0.1
        self.n_drone   = 3
        self.n_var     = 8
        self.m         = self.n_drone * self.n_var      # 24
        self.n         = 7                              # obs
        self.ratio_imu = 1                              # IMU chaque pas
        self.ratio_gps = 5                              # GPS+dist tous les 5 pas

        # --- bruits process (écarts-types par composante d'état) -----------
        sig_accel = 5e-2
        sig_autre = 1e-6
        w = np.full(self.m, sig_autre)
        for b in (0, 8, 16):
            w[b + 4] = w[b + 5] = sig_accel
        self.w_sigma = torch.tensor(w, dtype=torch.float32, device=device)

        # --- bruits mesure (R) ---------------------------------------------
        self.sig_gps = 0.5
        self.sig_acc = 0.1
        self.sig_d   = 0.5
        R_diag = [self.sig_gps**2, self.sig_gps**2,      # GPS x1,y1
                  self.sig_acc**2, self.sig_acc**2,      # IMU x2,y2
                  self.sig_d**2,  self.sig_d**2, self.sig_d**2]  # d12,d23,d13
        self.R = torch.diag(torch.tensor(R_diag, dtype=torch.float32, device=device))

        # --- Q (covariance process pour l'EKF) -----------------------------
        Q = np.eye(self.m) * 1e-3
        for b in (0, 8, 16):
            Q[b + 4, b + 4] = Q[b + 5, b + 5] = 0.5**2
            Q[b + 6, b + 6] = Q[b + 7, b + 7] = 1e-5**2
        Q[10, 10] = Q[11, 11] = 0.1**2     # accel drone 2 = random walk
        Q[12, 12] = Q[13, 13] = 1e-2**2
        self.Q = torch.tensor(Q, dtype=torch.float32, device=device)

        # --- Matrices F et B ----------------------------------------------
        self.F_true   = self._block_F(accel_const=(False, False, False))
        self.B_true   = self._build_B(estim_d2=True)
        # Le filtre suppose accel drone 2 = état (random walk) -> F avec a=1 sur drone2
        self.F_filter = self._block_F(accel_const=(False, True, False))
        self.B_filter = self._build_B(estim_d2=False)  # drone2 non commandé (random walk)

        # --- covariance initiale (P0) -------------------------------------
        sP_x, sP_v, sP_a, sP_b = 2.0, 0.5, 0.5, 1.0
        P0 = np.eye(self.m)
        for b in (0, 8, 16):
            P0[b, b]     = P0[b+1, b+1] = sP_x**2
            P0[b+2, b+2] = P0[b+3, b+3] = sP_v**2
            P0[b+4, b+4] = P0[b+5, b+5] = sP_a**2
            P0[b+6, b+6] = P0[b+7, b+7] = sP_b**2
        self.P0 = torch.tensor(P0, dtype=torch.float32, device=device)

        # --- état initial vrai --------------------------------------------
        x0 = np.concatenate(([0,  10, 1, 0, 0, 0, 0,    0],
                             [10,  0, 1, 0, 0, 0, 0.5, -0.2],
                             [0, -10, 1, 0, 0, 0, 0,    0])).astype(np.float32)
        self.x0 = torch.tensor(x0, dtype=torch.float32, device=device).reshape(self.m, 1)

        # --- priors pour init des hidden states de KalmanNet ---------------
        # (le réseau les utilise comme état caché initial, cf. init_hidden)
        self.prior_Q     = self.Q.clone()
        self.prior_Sigma = self.P0.clone()
        # S0 ~ H P0 H^T + R, on l'approxime par R (H dépend de l'état -> approx)
        self.prior_S     = self.R.clone()

    # ----- construction matricielle ----------------------------------------
    def _Fmat(self, accel_const):
        dt = self.dt
        a = 1.0 if accel_const else 0.0
        return np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0,         0, 0],
            [0, 1, 0, dt, 0,         0.5*dt*dt, 0, 0],
            [0, 0, 1, 0,  dt,        0,         0, 0],
            [0, 0, 0, 1,  0,         dt,        0, 0],
            [0, 0, 0, 0,  a,         0,         0, 0],
            [0, 0, 0, 0,  0,         a,         0, 0],
            [0, 0, 0, 0,  0,         0,         1, 0],
            [0, 0, 0, 0,  0,         0,         0, 1]], dtype=np.float32)

    def _block_F(self, accel_const):
        from scipy.linalg import block_diag
        F = block_diag(self._Fmat(accel_const[0]),
                       self._Fmat(accel_const[1]),
                       self._Fmat(accel_const[2]))
        return torch.tensor(F, dtype=torch.float32, device=self.device)

    def _Bmat(self, cols):
        M = np.zeros((8, 6), dtype=np.float32)
        M[4, cols[0]] = 1.0
        M[5, cols[1]] = 1.0
        return M

    def _build_B(self, estim_d2):
        # u est de dim 6 : (ax1,ay1, ax2,ay2, ax3,ay3) commandés
        b1 = self._Bmat([0, 1])
        b2 = self._Bmat([2, 3]) if estim_d2 else np.zeros((8, 6), np.float32)
        b3 = self._Bmat([4, 5])
        B = np.concatenate((b1, b2, b3), axis=0)
        return torch.tensor(B, dtype=torch.float32, device=self.device)

    # ----- dynamique (BATCHÉE, commandée) ----------------------------------
    def f(self, x, u, true=False):
        """
        Propagation de l'état. COMMANDÉE par u.
          x : [batch, m, 1]
          u : [batch, 6, 1]
        Retourne [batch, m, 1].
        """
        Fm = self.F_true if true else self.F_filter
        Bm = self.B_true if true else self.B_filter
        return torch.matmul(Fm, x) + torch.matmul(Bm, u)

    def h(self, x):
        """
        Observation (BATCHÉE, non-linéaire à cause des distances).
          x : [batch, m, 1]  ->  y : [batch, n=7, 1]
        Ordre : [GPSx1, GPSy1, IMUx2, IMUy2, d12, d23, d13]
        """
        b = x.shape[0]
        xs = x.squeeze(-1)                      # [batch, m]
        # positions
        x1, y1 = xs[:, 0],  xs[:, 1]
        x2, y2 = xs[:, 8],  xs[:, 9]
        x3, y3 = xs[:, 16], xs[:, 17]
        # IMU drone 2 : accel + biais
        imu_x = xs[:, 12] + xs[:, 14]
        imu_y = xs[:, 13] + xs[:, 15]
        # distances inter-drones
        eps = 1e-6                              # évite division par ~0 (NaN connu)
        d12 = torch.sqrt((x1 - x2)**2 + (y1 - y2)**2 + eps)
        d23 = torch.sqrt((x2 - x3)**2 + (y2 - y3)**2 + eps)
        d13 = torch.sqrt((x1 - x3)**2 + (y1 - y3)**2 + eps)
        y = torch.stack([x1, y1, imu_x, imu_y, d12, d23, d13], dim=1)  # [batch, 7]
        return y.unsqueeze(-1)                  # [batch, 7, 1]

    def obs_mask(self, step):
        """
        Masque [n] indiquant quelles composantes de y sont DISPONIBLES à ce pas.
        IMU (idx 2,3) toujours ; GPS+dist (idx 0,1,4,5,6) tous les ratio_gps pas.
        """
        m = torch.zeros(self.n, device=self.device)
        if step % self.ratio_imu == 0:
            m[2] = m[3] = 1.0
        if step % self.ratio_gps == 0:
            m[0] = m[1] = 1.0          # GPS
            m[4] = m[5] = m[6] = 1.0   # distances
        return m


# =============================================================================
# 2. GÉNÉRATION DE DONNÉES  (commandes u variées -> meilleure généralisation)
# =============================================================================
def build_command_sequence(T, dt, rng):
    """
    Génère une séquence de commandes u zero-mean variées (amplitudes/phases/freq
    aléatoires). Zero-mean impératif (sinon divergence position, cf. tes notes).
    Retourne u_seq : [T, 6, 1]   (ax1,ay1, ax2,ay2, ax3,ay3)
    """
    t = np.arange(1, T + 1) * dt
    u_seq = np.zeros((T, 6), dtype=np.float32)
    for d in range(3):                          # un bloc (ax,ay) par drone
        Ax, Ay = rng.uniform(0.5, 1.5, size=2)
        fx, fy = rng.uniform(0.3, 1.0, size=2)
        px, py = rng.uniform(0, 2*np.pi, size=2)
        u_seq[:, 2*d]   = Ax * np.cos(2*np.pi*fx*t + px)
        u_seq[:, 2*d+1] = Ay * np.sin(2*np.pi*fy*t + py)
    return torch.tensor(u_seq, dtype=torch.float32).unsqueeze(-1)  # [T,6,1]


def generate_trajectory(sm: SystemModel, rng, init_perturb=True, r_scale=1.0):
    """
    Génère UNE trajectoire (vérité + observations bruitées + commandes).
      r_scale : facteur multiplicatif sur l'écart-type de mesure (sqrt(R)).
                =1.0 -> bruit nominal ; sert au balayage MSE[dB] vs 1/r^2.
    Retourne :
      X   : [T+1, m, 1]   états vrais
      Y   : [T+1, n, 1]   observations bruitées (y[0] non utilisé)
      U   : [T,  6, 1]    commandes
      M   : [T+1, n]      masques de disponibilité capteur
    """
    T, m, n = CFG.T, sm.m, sm.n
    dev = sm.device
    U = build_command_sequence(T, sm.dt, rng).to(dev)
    sqrtR = torch.linalg.cholesky(sm.R) * r_scale

    X = torch.zeros(T + 1, m, 1, device=dev)
    Y = torch.zeros(T + 1, n, 1, device=dev)
    M = torch.zeros(T + 1, n, device=dev)

    # état initial (légèrement perturbé en position d'un run à l'autre)
    x = sm.x0.clone()
    if init_perturb:
        for b in (0, 8, 16):
            x[b:b+2, 0] += torch.tensor(rng.normal(0, 1.0, size=2), dtype=torch.float32, device=dev)
    X[0] = x

    for k in range(1, T + 1):
        u = U[k-1].unsqueeze(0)                              # [1,6,1]
        # process noise
        w = (torch.randn(m, 1, device=dev) * sm.w_sigma.reshape(m, 1))
        x = sm.f(x.unsqueeze(0), u, true=True).squeeze(0) + w
        X[k] = x
        # observation parfaite puis bruit
        y_clean = sm.h(x.unsqueeze(0)).squeeze(0)            # [n,1]
        v = torch.matmul(sqrtR, torch.randn(n, 1, device=dev))
        Y[k] = y_clean + v
        M[k] = sm.obs_mask(k)
    return X, Y, U, M


def generate_dataset(sm, n_traj, seed):
    rng = np.random.default_rng(seed)
    Xs, Ys, Us, Ms = [], [], [], []
    for _ in range(n_traj):
        X, Y, U, Mk = generate_trajectory(sm, rng)
        Xs.append(X); Ys.append(Y); Us.append(U); Ms.append(Mk)
    return (torch.stack(Xs), torch.stack(Ys),
            torch.stack(Us), torch.stack(Ms))   # [N,T+1,...]


# =============================================================================
# 3. EKF BASELINE  (fournit le couloir +-3 sigma de référence)
# =============================================================================
class EKF:
    """EKF batché minimal, mêmes f/h/Q/R que KalmanNet. Forme de Joseph."""

    def __init__(self, sm: SystemModel):
        self.sm = sm
        self.m, self.n = sm.m, sm.n
        self.dev = sm.device

    def _jac_h(self, x):
        """Jacobienne de h évaluée en x : [n, m] (batch=1 ici)."""
        sm = self.sm
        xs = x.squeeze(-1).squeeze(0)         # [m]
        H = torch.zeros(self.n, self.m, device=self.dev)
        # GPS drone 1
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        # IMU drone 2 (ax2+bx2, ay2+by2)
        H[2, 12] = 1.0; H[2, 14] = 1.0
        H[3, 13] = 1.0; H[3, 15] = 1.0
        # distances
        def fill(row, i, j):
            dx = xs[i] - xs[j]; dy = xs[i+1] - xs[j+1]
            d = torch.sqrt(dx**2 + dy**2 + 1e-9)
            H[row, i]   =  dx/d; H[row, i+1] =  dy/d
            H[row, j]   = -dx/d; H[row, j+1] = -dy/d
        fill(4, 0, 8)    # d12
        fill(5, 8, 16)   # d23
        fill(6, 0, 16)   # d13
        return H

    def run(self, Y, U, M, x0=None, P0=None):
        """
        Filtre une trajectoire. Y,U,M : [T+1,...] / [T,...].
        Retourne x_hist [T+1,m,1], P_hist [T+1,m,m].
        """
        sm = self.sm
        T = U.shape[0]
        x = (x0 if x0 is not None else sm.x0).clone().reshape(1, self.m, 1)
        P = (P0 if P0 is not None else sm.P0).clone()
        I = torch.eye(self.m, device=self.dev)

        x_hist = torch.zeros(T + 1, self.m, 1, device=self.dev)
        P_hist = torch.zeros(T + 1, self.m, self.m, device=self.dev)
        x_hist[0] = x.squeeze(0); P_hist[0] = P

        for k in range(1, T + 1):
            u = U[k-1].unsqueeze(0)
            # --- prédiction ---
            x = sm.f(x, u, true=False)
            F = sm.F_filter
            P = F @ P @ F.T + sm.Q
            # --- update masqué ---
            mask = M[k]                                 # [n]
            idx = torch.nonzero(mask > 0).squeeze(-1)
            if idx.numel() > 0:
                Hf = self._jac_h(x)[idx]                # [k_obs, m]
                yhat = sm.h(x).squeeze(0)[idx]          # [k_obs,1]
                ymeas = Y[k][idx]                       # [k_obs,1]
                Rk = sm.R[idx][:, idx]
                innov = ymeas - yhat
                S = Hf @ P @ Hf.T + Rk
                K = P @ Hf.T @ torch.linalg.inv(S)
                # innov:[k_obs,1]  K:[m,k_obs]  ->  correction:[m,1]
                x = x + (K @ innov).reshape(1, self.m, 1)
                A = I - K @ Hf
                P = A @ P @ A.T + K @ Rk @ K.T
            x_hist[k] = x.squeeze(0); P_hist[k] = P
        return x_hist, P_hist


# =============================================================================
# 4. KALMANNET — réseau de gain de Kalman (DEUX ARCHITECTURES)
# =============================================================================
class KalmanNetNN(nn.Module):
    """
    archi='archi1' : 1 GRU, features {F2 (innov diff), F4 (fw update diff)}
    archi='archi2' : 3 GRU cascade Q->Sigma->S, toutes les features F1..F4
    Système COMMANDÉ : on stocke u courant pour step_prior.
    """

    def __init__(self, sm: SystemModel, archi="archi2",
                 in_mult=CFG.IN_MULT, out_mult=CFG.OUT_MULT):
        super().__init__()
        self.sm = sm
        self.archi = archi
        self.m, self.n = sm.m, sm.n
        self.dev = sm.device
        self.in_mult, self.out_mult = in_mult, out_mult
        self.f = sm.f
        self.h = sm.h
        self.prior_Q     = sm.prior_Q
        self.prior_Sigma = sm.prior_Sigma
        self.prior_S     = sm.prior_S
        if archi == "archi1":
            self._build_archi1()
        else:
            self._build_archi2()
        self.to(self.dev)

    # ---------------------------------------------------------------- archi 1
    def _build_archi1(self):
        m, n = self.m, self.n
        # features : F2 (dim n) + F4 (dim m)  -> hidden GRU ~ 10*(m^2+n^2)
        d_in  = n + m
        self.h_dim = 10 * (m**2 + n**2)
        self.fc_in = nn.Sequential(nn.Linear(d_in, d_in * self.in_mult), nn.ReLU())
        self.gru   = nn.GRU(d_in * self.in_mult, self.h_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(self.h_dim, (m * n) * 4), nn.ReLU(),
            nn.Linear((m * n) * 4, m * n))

    # ---------------------------------------------------------------- archi 2
    def _build_archi2(self):
        m, n, im, om = self.m, self.n, self.in_mult, self.out_mult
        self.d_hidden_Q     = m * m
        self.d_hidden_Sigma = m * m
        self.d_hidden_S     = n * n
        # GRU Q  <- FC5(F4)
        self.FC5   = nn.Sequential(nn.Linear(m, m * im), nn.ReLU())
        self.GRU_Q = nn.GRU(m * im, self.d_hidden_Q)
        # GRU Sigma <- [out_Q, FC6(F3)]
        self.FC6   = nn.Sequential(nn.Linear(m, m * im), nn.ReLU())
        self.GRU_Sigma = nn.GRU(self.d_hidden_Q + m * im, self.d_hidden_Sigma)
        # GRU S  <- [FC1(Sigma), FC7(F1,F2)]
        self.FC1   = nn.Sequential(nn.Linear(self.d_hidden_Sigma, n*n), nn.ReLU())
        self.FC7   = nn.Sequential(nn.Linear(2*n, 2*n*im), nn.ReLU())
        self.GRU_S = nn.GRU(n*n + 2*n*im, self.d_hidden_S)
        # sorties : KG = FC2([Sigma,S]) ; backward FC3,FC4
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_hidden_S + self.d_hidden_Sigma,
                      (self.d_hidden_S + self.d_hidden_Sigma) * om), nn.ReLU(),
            nn.Linear((self.d_hidden_S + self.d_hidden_Sigma) * om, n * m))
        self.FC3 = nn.Sequential(nn.Linear(self.d_hidden_S + n*m, m*m), nn.ReLU())
        self.FC4 = nn.Sequential(nn.Linear(self.d_hidden_Sigma + m*m,
                                           self.d_hidden_Sigma), nn.ReLU())

    # -------------------------------------------------- init séquence / hidden
    def init_sequence(self, x0, batch):
        """x0 : [m,1]. Initialise les estimés + hidden states pour 'batch' traj."""
        self.batch = batch
        self.m1x_post      = x0.reshape(1, self.m, 1).repeat(batch, 1, 1).to(self.dev)
        self.m1x_post_prev = self.m1x_post.clone()
        self.m1x_prior_prev = self.m1x_post.clone()
        self.y_prev = self.h(self.m1x_post)
        self._init_hidden()

    def _init_hidden(self):
        b = self.batch
        if self.archi == "archi1":
            self.hid = torch.zeros(1, b, self.h_dim, device=self.dev)
        else:
            self.h_Q     = self.prior_Q.flatten().reshape(1,1,-1).repeat(1,b,1).to(self.dev)
            self.h_Sigma = self.prior_Sigma.flatten().reshape(1,1,-1).repeat(1,b,1).to(self.dev)
            self.h_S     = self.prior_S.flatten().reshape(1,1,-1).repeat(1,b,1).to(self.dev)

    # -------------------------------------------------- étape de prédiction
    def step_prior(self, u):
        self.m1x_prior = self.f(self.m1x_post, u, true=False)   # COMMANDE ici
        self.m1y       = self.h(self.m1x_prior)

    # -------------------------------------------------- features + gain
    def _features(self, y):
        f2 = (y - self.m1y).squeeze(-1)                          # innovation diff
        f4 = (self.m1x_post - self.m1x_prior_prev).squeeze(-1)   # fw update diff
        f1 = (y - self.y_prev).squeeze(-1)                       # obs diff
        f3 = (self.m1x_post - self.m1x_post_prev).squeeze(-1)    # fw evol diff
        norm = lambda v: F.normalize(v, p=2, dim=1, eps=1e-12)
        return norm(f1), norm(f2), norm(f3), norm(f4)

    def _kgain_archi1(self, f2, f4):
        x = torch.cat([f2, f4], dim=1).unsqueeze(0)             # [1,b,n+m]
        x = self.fc_in(x)
        out, self.hid = self.gru(x, self.hid)
        kg = self.fc_out(out)                                   # [1,b,m*n]
        return kg.reshape(self.batch, self.m, self.n)

    def _kgain_archi2(self, f1, f2, f3, f4):
        f1=f1.unsqueeze(0); f2=f2.unsqueeze(0); f3=f3.unsqueeze(0); f4=f4.unsqueeze(0)
        out_FC5 = self.FC5(f4)
        out_Q, self.h_Q = self.GRU_Q(out_FC5, self.h_Q)
        out_FC6 = self.FC6(f3)
        in_Sigma = torch.cat([out_Q, out_FC6], dim=2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)
        out_FC1 = self.FC1(out_Sigma)
        out_FC7 = self.FC7(torch.cat([f1, f2], dim=2))
        in_S = torch.cat([out_FC1, out_FC7], dim=2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)
        out_FC2 = self.FC2(torch.cat([out_Sigma, out_S], dim=2))   # KG
        # backward : met à jour h_Sigma
        out_FC3 = self.FC3(torch.cat([out_S, out_FC2], dim=2))
        out_FC4 = self.FC4(torch.cat([out_Sigma, out_FC3], dim=2))
        self.h_Sigma = out_FC4
        return out_FC2.reshape(self.batch, self.m, self.n)

    # -------------------------------------------------- un pas KalmanNet
    def step(self, y, u, mask):
        """
        y    : [batch, n, 1]
        u    : [batch, 6, 1]
        mask : [n]  (1 = obs dispo, 0 = absente -> innovation masquée)
        """
        self.step_prior(u)
        f1, f2, f3, f4 = self._features(y)
        if self.archi == "archi1":
            KG = self._kgain_archi1(f2, f4)
        else:
            KG = self._kgain_archi2(f1, f2, f3, f4)
        self.KGain = KG
        # innovation MASQUÉE : on annule les canaux absents
        dy = (y - self.m1y) * mask.reshape(1, self.n, 1)
        inov = torch.bmm(KG, dy)                                # [batch,m,1]
        self.m1x_post_prev  = self.m1x_post
        self.m1x_post       = self.m1x_prior + inov
        self.m1x_prior_prev = self.m1x_prior
        self.y_prev = y
        return self.m1x_post

    def forward(self, y, u, mask):
        return self.step(y, u, mask)


# =============================================================================
# 5. ENTRAÎNEMENT
# =============================================================================
def train(sm, model, data_train, data_val, tag="archi2"):
    Xtr, Ytr, Utr, Mtr = data_train
    Xva, Yva, Uva, Mva = data_val
    N, Tp1 = Xtr.shape[0], Xtr.shape[1]
    T = Tp1 - 1

    opt = torch.optim.Adam(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)
    mse = nn.MSELoss(reduction='mean')

    hist_train, hist_val = [], []
    best_val = float('inf')
    best_path = os.path.join(CFG.OUT_DIR, f"knet_{tag}.pt")

    for epoch in range(CFG.N_EPOCHS):
        # ------------------- TRAIN -------------------
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = N // CFG.N_BATCH
        for bi in range(n_batches):
            idx = perm[bi*CFG.N_BATCH:(bi+1)*CFG.N_BATCH]
            Xb, Yb, Ub, Mb = Xtr[idx], Ytr[idx], Utr[idx], Mtr[idx]
            b = Xb.shape[0]
            model.init_sequence(sm.x0, b)
            loss = 0.0
            for k in range(1, T + 1):
                y = Yb[:, k]                      # [b,n,1]
                u = Ub[:, k-1]                    # [b,6,1]
                mask = Mb[0, k]                   # même cadence pour tous
                xhat = model(y, u, mask)
                loss = loss + mse(xhat, Xb[:, k])
            loss = loss / T
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
            opt.step()
            epoch_loss += loss.item()
        hist_train.append(epoch_loss / max(n_batches, 1))

        # ------------------- VAL -------------------
        model.eval()
        with torch.no_grad():
            b = Xva.shape[0]
            model.init_sequence(sm.x0, b)
            vloss = 0.0
            for k in range(1, T + 1):
                xhat = model(Yva[:, k], Uva[:, k-1], Mva[0, k])
                vloss = vloss + mse(xhat, Xva[:, k]).item()
            vloss /= T
        hist_val.append(vloss)

        if vloss < best_val:
            best_val = vloss
            torch.save({'state_dict': model.state_dict(),
                        'archi': model.archi}, best_path)

        if epoch % 5 == 0 or epoch == CFG.N_EPOCHS - 1:
            print(f"[{tag}] epoch {epoch:3d} | train {hist_train[-1]:.4e} "
                  f"| val {vloss:.4e} | best {best_val:.4e}")

    print(f"[{tag}] modèle sauvé -> {best_path}")
    return hist_train, hist_val, best_path


# =============================================================================
# 6. INFÉRENCE (une trajectoire)
# =============================================================================
@torch.no_grad()
def run_knet(sm, model, Y, U, M):
    T = U.shape[0]
    model.eval()
    model.init_sequence(sm.x0, 1)
    xh = torch.zeros(T + 1, sm.m, 1, device=sm.device)
    xh[0] = sm.x0
    for k in range(1, T + 1):
        y = Y[k].unsqueeze(0)
        u = U[k-1].unsqueeze(0)
        xh[k] = model(y, u, M[k]).squeeze(0)
    return xh


# =============================================================================
# 7. MONTE-CARLO (togglable)  -> sigma empirique KalmanNet
# =============================================================================
def monte_carlo_knet(sm, model, seed=777, n_mc=CFG.N_MC):
    """
    Lance n_mc trajectoires, renvoie l'écart-type empirique de l'erreur KNet
    par composante et par instant : [T+1, m]. Sert de couloir empirique.
    Si MODE_MONTE_CARLO=False on ne l'appelle pas (le reste tourne pareil).
    """
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n_mc):
        X, Y, U, Mk = generate_trajectory(sm, rng)
        xh = run_knet(sm, model, Y, U, Mk)
        errs.append((xh - X).squeeze(-1).cpu().numpy())   # [T+1,m]
    errs = np.stack(errs)                                   # [n_mc,T+1,m]
    sigma_emp = errs.std(axis=0)                            # [T+1,m]
    return sigma_emp


# =============================================================================
# 8. PLOTS
# =============================================================================
LABELS = ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'bx', 'by']
BASES  = {1: 0, 2: 8, 3: 16}


def plot_loss(hist_train, hist_val, tag, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist_train, label="Loss entraînement", lw=2)
    ax.plot(hist_val,   label="Loss validation", lw=2)
    ax.set_yscale('log')
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (log)")
    ax.set_title(f"Courbe d'apprentissage KalmanNet — {tag}")
    ax.grid(True, ls=':', alpha=0.7); ax.legend()
    fig.tight_layout()
    p = os.path.join(outdir, f"loss_{tag}.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def plot_drone(sm, drone, X, x_ekf, x_knet, P_ekf, temps, tag,
               sigma_mc=None, outdir=CFG.OUT_DIR):
    """
    Figure par drone : 4x2 sous-plots (x,y,vx,vy,ax,ay,bx,by).
    Affiche l'ERREUR (estimé - vrai) pour EKF et KalmanNet,
    + couloir +-3sigma de l'EKF (réf.) [+ couloir MC empirique KNet si fourni].
    """
    base = BASES[drone]
    fig, axs = plt.subplots(4, 2, figsize=(13, 9), sharex=True)
    axs = axs.flatten()
    for i in range(8):
        idx = base + i
        err_ekf  = (x_ekf[:, idx, 0]  - X[:, idx, 0]).cpu().numpy()
        err_knet = (x_knet[:, idx, 0] - X[:, idx, 0]).cpu().numpy()
        sig_ekf  = np.sqrt(P_ekf[:, idx, idx].cpu().numpy())
        axs[i].fill_between(temps, -3*sig_ekf, 3*sig_ekf, color='blue', alpha=0.15,
                            label=r'$\pm 3\sigma$ EKF')
        if sigma_mc is not None:
            s = sigma_mc[:, idx]
            axs[i].plot(temps,  3*s, color='purple', ls='--', lw=1, alpha=0.8,
                        label=r'$\pm 3\sigma$ KNet (MC)')
            axs[i].plot(temps, -3*s, color='purple', ls='--', lw=1, alpha=0.8)
        axs[i].plot(temps, err_ekf,  color='green', lw=1.3, label='Erreur EKF')
        axs[i].plot(temps, err_knet, color='red',   lw=1.3, label='Erreur KalmanNet')
        axs[i].axhline(0, color='k', lw=0.6)
        axs[i].set_title(f"{LABELS[i]} : estimé − vrai", fontsize=10)
        axs[i].grid(True, ls=':', alpha=0.7)
    axs[6].set_xlabel("Temps (s)"); axs[7].set_xlabel("Temps (s)")
    fig.suptitle(f"Drone {drone} — erreurs EKF vs KalmanNet ({tag})",
                 fontsize=13, fontweight='bold')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(outdir, f"drone{drone}_{tag}.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


# =============================================================================
# 8bis. MÉTRIQUES INSPIRÉES DU PAPIER (togglables)
# =============================================================================
def mse_db_curve(sm, model, r_levels_db, n_mc, seed=2024, ekf=None):
    """
    Courbe MSE[dB] vs 1/r^2[dB] (Fig.5-9 du papier). Pour chaque niveau de bruit
    mesure, génère n_mc trajectoires, filtre avec KalmanNet (et EKF si fourni),
    et renvoie la MSE moyenne (sur l'état complet) en dB.
    1/r^2 [dB] = -20*log10(r_scale)  (r_scale=1 -> 0 dB de référence).
    """
    knet_db, ekf_db = [], []
    for r_db in r_levels_db:
        r_scale = 10 ** (-r_db / 20.0)        # 1/r^2[dB] -> facteur sur sqrt(R)
        rng = np.random.default_rng(seed + int(r_db))
        mse_k, mse_e = [], []
        for _ in range(n_mc):
            X, Y, U, Mk = generate_trajectory(sm, rng, r_scale=r_scale)
            xh = run_knet(sm, model, Y, U, Mk)
            mse_k.append(((xh - X)**2).mean().item())
            if ekf is not None:
                xe, _ = ekf.run(Y, U, Mk)
                mse_e.append(((xe - X)**2).mean().item())
        knet_db.append(10*np.log10(np.mean(mse_k)))
        if ekf is not None:
            ekf_db.append(10*np.log10(np.mean(mse_e)))
    return np.array(r_levels_db), np.array(knet_db), (np.array(ekf_db) if ekf else None)


def plot_mse_db(r_db, knet_db, ekf_db, tag, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_db, knet_db, 'o-', color='red', lw=2, label='KalmanNet')
    if ekf_db is not None:
        ax.plot(r_db, ekf_db, 's--', color='green', lw=2, label='EKF (MMSE réf.)')
    ax.set_xlabel(r'$1/r^2$ [dB]'); ax.set_ylabel('MSE [dB]')
    ax.set_title(f"MSE[dB] vs niveau de bruit — {tag}")
    ax.grid(True, ls=':', alpha=.7); ax.legend()
    fig.tight_layout()
    p = os.path.join(outdir, f"mse_db_{tag}.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def nci_coverage(sm, model, P_ekf, seed=3030, n_mc=CFG.N_MC):
    """
    Cohérence statistique du couloir. On utilise la covariance EKF comme
    référence de confiance (KNet n'a pas de P natif) et on mesure :
      - NCI moyen : moyenne de e^T P^-1 e / m  (idéal ~ 1)
      - taux de couverture 3sigma : fraction des erreurs dans +-3sigma_EKF
    Renvoie (nci_t [T+1], coverage_t [T+1]).
    """
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n_mc):
        X, Y, U, Mk = generate_trajectory(sm, rng)
        xh = run_knet(sm, model, Y, U, Mk)
        errs.append((xh - X).squeeze(-1).cpu().numpy())     # [T+1,m]
    errs = np.stack(errs)                                    # [n_mc,T+1,m]
    Pnp = P_ekf.cpu().numpy()                                # [T+1,m,m]
    Tp1, m = errs.shape[1], errs.shape[2]
    nci = np.zeros(Tp1); cover = np.zeros(Tp1)
    for k in range(Tp1):
        Pk = Pnp[k] + 1e-9*np.eye(m)
        Pinv = np.linalg.inv(Pk)
        sig = np.sqrt(np.diag(Pk))
        ek = errs[:, k, :]                                   # [n_mc,m]
        nci[k]   = np.mean(np.einsum('ij,jk,ik->i', ek, Pinv, ek)) / m
        cover[k] = np.mean(np.all(np.abs(ek) <= 3*sig, axis=1))
    return nci, cover


def plot_nci(temps, nci, cover, tag, outdir):
    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axs[0].plot(temps, nci, color='purple', lw=1.6)
    axs[0].axhline(1.0, color='k', ls='--', lw=1, label='NCI idéal = 1')
    axs[0].axhline(2.0, color='r', ls=':', lw=1, label='seuil 2 (overconfidence)')
    axs[0].set_ylabel('NCI moyen'); axs[0].set_title(f"Cohérence statistique — {tag}")
    axs[0].grid(True, ls=':', alpha=.7); axs[0].legend()
    axs[1].plot(temps, 100*cover, color='teal', lw=1.6)
    axs[1].axhline(99.7, color='k', ls='--', lw=1, label='cible 99.7% (3σ)')
    axs[1].set_ylabel('Couverture 3σ [%]'); axs[1].set_xlabel('Temps (s)')
    axs[1].grid(True, ls=':', alpha=.7); axs[1].legend()
    fig.tight_layout()
    p = os.path.join(outdir, f"nci_{tag}.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p



def main():
    os.makedirs(CFG.OUT_DIR, exist_ok=True)
    torch.manual_seed(CFG.SEED); np.random.seed(CFG.SEED)
    sm = SystemModel()

    print("== Génération des données ==")
    data_train = generate_dataset(sm, CFG.N_TRAIN, seed=CFG.SEED)
    data_val   = generate_dataset(sm, CFG.N_VAL,   seed=CFG.SEED + 1)

    # trajectoire de test (référence des plots détaillés)
    rng_test = np.random.default_rng(CFG.SEED + 99)
    Xte, Yte, Ute, Mte = generate_trajectory(sm, rng_test)
    temps = (np.arange(CFG.T + 1) * sm.dt)

    # EKF baseline sur la trajectoire de test
    print("== EKF baseline ==")
    ekf = EKF(sm)
    x_ekf, P_ekf = ekf.run(Yte, Ute, Mte)

    archis = (["archi1", "archi2"] if CFG.ARCHI_TO_TRAIN == "both"
              else [CFG.ARCHI_TO_TRAIN])
    produced = []

    for archi in archis:
        print(f"\n===== ENTRAÎNEMENT {archi} =====")
        model = KalmanNetNN(sm, archi=archi)
        ht, hv, ckpt = train(sm, model, data_train, data_val, tag=archi)
        produced.append(plot_loss(ht, hv, archi, CFG.OUT_DIR))

        # recharge le meilleur modèle
        state = torch.load(ckpt, map_location=sm.device)
        model.load_state_dict(state['state_dict'])

        # inférence test
        x_knet = run_knet(sm, model, Yte, Ute, Mte)

        # Monte-Carlo (togglable)
        sigma_mc = None
        if CFG.MODE_MONTE_CARLO:
            print(f"  Monte-Carlo {CFG.N_MC} runs ({archi})...")
            sigma_mc = monte_carlo_knet(sm, model)

        # plots par drone
        for d in (1, 2, 3):
            produced.append(plot_drone(sm, d, Xte, x_ekf, x_knet, P_ekf,
                                       temps, archi, sigma_mc=sigma_mc))

        # MSE position drone 2
        mse_pos = ((x_knet[:, 8:10, 0] - Xte[:, 8:10, 0])**2).mean().item()
        mse_ekf = ((x_ekf[:, 8:10, 0]  - Xte[:, 8:10, 0])**2).mean().item()
        print(f"  MSE pos drone2 : KNet={mse_pos:.4f} | EKF={mse_ekf:.4f}")

        # --- métrique MSE[dB] vs 1/r^2 (togglable) -----------------------
        if CFG.PLOT_MSE_DB:
            print(f"  Courbe MSE[dB] ({archi})...")
            r_db, k_db, e_db = mse_db_curve(sm, model, CFG.R_SWEEP_DB,
                                            CFG.N_MC_DB, ekf=ekf)
            produced.append(plot_mse_db(r_db, k_db, e_db, archi, CFG.OUT_DIR))

        # --- métrique NCI / couverture 3sigma (togglable, requiert MC) ---
        if CFG.PLOT_NCI:
            print(f"  NCI / couverture 3σ ({archi})...")
            nci, cover = nci_coverage(sm, model, P_ekf)
            produced.append(plot_nci(temps, nci, cover, archi, CFG.OUT_DIR))

    print("\n== Figures générées ==")
    for p in produced:
        print("  ", p)
    return produced


if __name__ == "__main__":
    main()
