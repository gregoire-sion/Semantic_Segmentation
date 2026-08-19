
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)


class CFG:

    ARCHI_TO_TRAIN = "both"

    MODE_MONTE_CARLO = False
    N_MC             = 50

    PLOT_MSE_DB = True
    R_SWEEP_DB = [-10, -5, 0, 5, 10, 20, 30]
    N_MC_DB = 30
    PLOT_NCI = True
    TRAIN_NOISE_SWEEP = True
    TRAIN_NOISE_DB = (-10, 30)

    INIT_OFFSET_P0 = True
    INIT_OFFSET_SCALE = 1.0

    USE_SAVED_DATASET = False
    DATASET_PATH = "./Entrainement_avec_perturbation_initiale/dataset.npz"

    N_TRAIN   = 200
    N_VAL     = 40
    N_TEST    = 10
    T         = 160
    SEED      = 42

    N_EPOCHS   = 40
    N_BATCH    = 32
    LR         = 3e-4
    WD         = 1e-4
    GRAD_CLIP  = 1.0
    TBPTT      = 20

    IN_MULT   = 5
    OUT_MULT  = 40

    # ---------------- Généralisation OOD ----------------
    # False -> modèle A : distribution étroite d'origine
    #          (A ~ U(0.9,1.1), bornes de phase figées à T/3 et 2T/3).
    # True  -> modèle B : randomisation de domaine sur la commande u.
    # Seule la commande change ; f, h, Q, R et la géométrie sont intacts.
    TRAIN_CMD_RANDOMIZE = False
    TRAIN_CMD_FAMILIES  = ("phases3_rand", "ou")

    OUT_DIR   = "./Dataset_ood" if TRAIN_CMD_RANDOMIZE else "./Dataset"
    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SystemModel:
    def __init__(self, device=CFG.DEVICE):
        self.device = device
        self.dt = 0.1
        self.n_drone = 3
        self.n_var = 8
        self.m = self.n_drone * self.n_var
        self.n = 7
        self.ratio_imu = 1
        self.ratio_gps = 5

        sig_accel = 5e-2
        sig_autre = 1e-6
        w = np.full(self.m, sig_autre)
        for b in (0, 8, 16):
            w[b + 4] = w[b + 5] = sig_accel
        self.w_sigma = torch.tensor(w, dtype=torch.float32, device=device)

        self.sig_gps = 0.5
        self.sig_acc = 0.01
        self.sig_d   = 0.5
        sig_R_gps, sig_R_acc, sig_R_d = 0.5, 0.1, 0.5
        R_diag = [sig_R_gps**2, sig_R_gps**2,
                  sig_R_acc**2, sig_R_acc**2,
                  sig_R_d**2,  sig_R_d**2, sig_R_d**2]
        self.R = torch.diag(torch.tensor(R_diag, dtype=torch.float32, device=device))
        Rgen_diag = [self.sig_gps**2, self.sig_gps**2,
                     self.sig_acc**2, self.sig_acc**2,
                     self.sig_d**2,  self.sig_d**2, self.sig_d**2]
        self.R_gen = torch.diag(torch.tensor(Rgen_diag, dtype=torch.float32, device=device))

        Q = np.eye(self.m) * 1e-3
        for b in (0, 8, 16):
            Q[b + 4, b + 4] = Q[b + 5, b + 5] = 0.5**2
            Q[b + 6, b + 6] = Q[b + 7, b + 7] = 1e-5**2
        Q[10, 10] = Q[11, 11] = 0.1**2
        Q[12, 12] = Q[13, 13] = 0.5**2
        self.Q = torch.tensor(Q, dtype=torch.float32, device=device)

        self.F_true   = self._block_F(accel_const=(False, False, False))
        self.B_true   = self._build_B(estim_d2=True)
        self.F_filter = self._block_F(accel_const=(False, True, False))
        self.B_filter = self._build_B(estim_d2=False)

        sP_x, sP_v, sP_a, sP_b = 2.0, 0.5, 0.5, 1.0
        P0 = np.eye(self.m)
        for b in (0, 8, 16):
            P0[b, b]     = P0[b+1, b+1] = sP_x**2
            P0[b+2, b+2] = P0[b+3, b+3] = sP_v**2
            P0[b+4, b+4] = P0[b+5, b+5] = sP_a**2
            P0[b+6, b+6] = P0[b+7, b+7] = sP_b**2
        self.P0 = torch.tensor(P0, dtype=torch.float32, device=device)

        x0 = np.concatenate(([0,  10, 1, 0, 0, 0, 0,    0],
                             [10,  0, 1, 0, 0, 0, 0.5, -0.2],
                             [0, -10, 1, 0, 0, 0, 0,    0])).astype(np.float32)
        self.x0 = torch.tensor(x0, dtype=torch.float32, device=device).reshape(self.m, 1)

        self.prior_Q     = self.Q.clone()
        self.prior_Sigma = self.P0.clone()
        self.prior_S     = self.R.clone()

        scale = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5])
        w_loss = np.tile(1.0 / scale**2, 3).astype(np.float32)
        self.loss_w = torch.tensor(w_loss, dtype=torch.float32, device=device).reshape(1, self.m, 1)

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
        b1 = self._Bmat([0, 1])
        b2 = self._Bmat([2, 3]) if estim_d2 else np.zeros((8, 6), np.float32)
        b3 = self._Bmat([4, 5])
        B = np.concatenate((b1, b2, b3), axis=0)
        return torch.tensor(B, dtype=torch.float32, device=self.device)

    def f(self, x, u, true=False):
        Fm = self.F_true if true else self.F_filter
        Bm = self.B_true if true else self.B_filter
        return torch.matmul(Fm, x) + torch.matmul(Bm, u)

    def h(self, x):
        b = x.shape[0]
        xs = x.squeeze(-1)
        x1, y1 = xs[:, 0],  xs[:, 1]
        x2, y2 = xs[:, 8],  xs[:, 9]
        x3, y3 = xs[:, 16], xs[:, 17]
        imu_x = xs[:, 12] + xs[:, 14]
        imu_y = xs[:, 13] + xs[:, 15]
        eps = 1e-6
        d12 = torch.sqrt((x1 - x2)**2 + (y1 - y2)**2 + eps)
        d23 = torch.sqrt((x2 - x3)**2 + (y2 - y3)**2 + eps)
        d13 = torch.sqrt((x1 - x3)**2 + (y1 - y3)**2 + eps)
        y = torch.stack([x1, y1, imu_x, imu_y, d12, d23, d13], dim=1)
        return y.unsqueeze(-1)

    def obs_mask(self, step):
        m = torch.zeros(self.n, device=self.device)
        if step % self.ratio_imu == 0:
            m[2] = m[3] = 1.0
        if step % self.ratio_gps == 0:
            m[0] = m[1] = 1.0
            m[4] = m[5] = m[6] = 1.0
        return m


def build_command_sequence(T, dt, rng):
    if getattr(CFG, "TRAIN_CMD_RANDOMIZE", False):
        from ood_commands import build_command, sample_train_family
        return build_command(T, dt, rng,
                             kind=sample_train_family(rng, CFG.TRAIN_CMD_FAMILIES))

    u_seq = np.zeros((T, 6), dtype=np.float32)
    A   = rng.uniform(0.9, 1.1)
    px0 = rng.uniform(0, 2*np.pi)
    py0 = rng.uniform(0, 2*np.pi)
    phi_x, phi_y = px0, py0
    for k in range(T):
        if k < T / 3:
            u_seq[k] = [A, 0., A, 0., A, 0.]
        elif k < 2 * T / 3:
            phi_x += 5 * dt
            phi_y += 1 * dt
            cx, sy = A*np.cos(phi_x), A*np.sin(phi_y)
            u_seq[k] = [cx, sy, cx, sy, cx, sy]
        else:
            u_seq[k] = [A, 0., A, 0., A, 0.]
    return torch.tensor(u_seq, dtype=torch.float32).unsqueeze(-1)


def build_command_ood(T, dt, rng, kind="3phases"):
    u_seq = np.zeros((T, 6), dtype=np.float32)
    if kind == "3phases":
        phi_x = phi_y = 0.0
        for k in range(T):
            if k < T / 3:
                u_seq[k] = [1., 0., 1., 0., 1., 0.]
            elif k < 2 * T / 3:
                phi_x += 5 * dt; phi_y += 1 * dt
                cx, sy = np.cos(phi_x), np.sin(phi_y)
                u_seq[k] = [cx, sy, cx, sy, cx, sy]
            else:
                u_seq[k] = [1., 0., 1., 0., 1., 0.]
    elif kind == "brutal":
        seg = max(1, T // 5)
        for k in range(T):
            s = (k // seg) % 4
            amp = rng.uniform(1.5, 2.5)
            table = {0: [amp, 0], 1: [0, amp], 2: [-amp, 0], 3: [0, -amp]}
            ax, ay = table[s]
            u_seq[k] = [ax, ay, ax, ay, ax, ay]
    return torch.tensor(u_seq, dtype=torch.float32).unsqueeze(-1)


def generate_trajectory(sm, rng, init_perturb=True, r_scale=1.0, u_seq=None):
    T, m, n = CFG.T, sm.m, sm.n
    dev = sm.device
    if u_seq is None:
        U = build_command_sequence(T, sm.dt, rng).to(dev)
    else:
        U = u_seq.to(dev) if hasattr(u_seq, "to") else torch.tensor(u_seq, dtype=torch.float32, device=dev)
    sqrtR = torch.linalg.cholesky(sm.R_gen) * r_scale

    X = torch.zeros(T + 1, m, 1, device=dev)
    Y = torch.zeros(T + 1, n, 1, device=dev)
    M = torch.zeros(T + 1, n, device=dev)

    x = sm.x0.clone()
    if init_perturb:
        if CFG.INIT_OFFSET_P0:
            L = torch.linalg.cholesky(sm.P0)
            xi = torch.tensor(rng.normal(0, 1, size=m), dtype=torch.float32, device=dev).reshape(m, 1)
            x = x + CFG.INIT_OFFSET_SCALE * (L @ xi)
        else:
            for b in (0, 8, 16):
                x[b:b+2, 0] += torch.tensor(rng.normal(0, 2.0, size=2), dtype=torch.float32, device=dev)
    X[0] = x

    for k in range(1, T + 1):
        u = U[k-1].unsqueeze(0)
        w = (torch.randn(m, 1, device=dev) * sm.w_sigma.reshape(m, 1))
        x = sm.f(x.unsqueeze(0), u, true=True).squeeze(0) + w
        X[k] = x
        y_clean = sm.h(x.unsqueeze(0)).squeeze(0)
        v = torch.matmul(sqrtR, torch.randn(n, 1, device=dev))
        Y[k] = y_clean + v
        M[k] = sm.obs_mask(k)
    return X, Y, U, M


def generate_dataset(sm, n_traj, seed, noise_sweep=False):
    rng = np.random.default_rng(seed)
    Xs, Ys, Us, Ms = [], [], [], []
    lo, hi = CFG.TRAIN_NOISE_DB
    for _ in range(n_traj):
        if noise_sweep:
            r_db = rng.uniform(lo, hi)
            r_scale = 10**(-r_db/20.0)
        else:
            r_scale = 1.0
        X, Y, U, Mk = generate_trajectory(sm, rng, r_scale=r_scale)
        Xs.append(X); Ys.append(Y); Us.append(U); Ms.append(Mk)
    return (torch.stack(Xs), torch.stack(Ys),
            torch.stack(Us), torch.stack(Ms))


def save_dataset(data_train, data_val, data_test, path=None):
    if path is None:
        path = CFG.DATASET_PATH or os.path.join(CFG.OUT_DIR, "dataset.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Xtr, Ytr, Utr, Mtr = data_train
    Xva, Yva, Uva, Mva = data_val
    Xte, Yte, Ute, Mte = data_test
    np.savez(path,
             Xtr=Xtr.cpu().numpy(), Ytr=Ytr.cpu().numpy(),
             Utr=Utr.cpu().numpy(), Mtr=Mtr.cpu().numpy(),
             Xva=Xva.cpu().numpy(), Yva=Yva.cpu().numpy(),
             Uva=Uva.cpu().numpy(), Mva=Mva.cpu().numpy(),
             Xte=Xte.cpu().numpy(), Yte=Yte.cpu().numpy(),
             Ute=Ute.cpu().numpy(), Mte=Mte.cpu().numpy())
    print(f">> Dataset sauvegardé -> {path}")
    print(f"   train={Xtr.shape[0]}  val={Xva.shape[0]}  test={Xte.shape[0]}")
    return path


def load_dataset(sm, path=None):
    if path is None:
        path = CFG.DATASET_PATH or os.path.join(CFG.OUT_DIR, "dataset.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} introuvable. Relance avec USE_SAVED_DATASET=False.")
    d = np.load(path)

    def to(k):
        return torch.tensor(d[k], dtype=torch.float32, device=sm.device)

    data_train = (to("Xtr"), to("Ytr"), to("Utr"), to("Mtr"))
    data_val   = (to("Xva"), to("Yva"), to("Uva"), to("Mva"))
    Xte = to("Xte")[0]; Yte = to("Yte")[0]
    Ute = to("Ute")[0]; Mte = to("Mte")[0]
    print(f">> Dataset chargé : {path}")
    print(f"   train={data_train[0].shape[0]}  val={data_val[0].shape[0]}  test={to('Xte').shape[0]} (1 utilisée)")
    return data_train, data_val, (Xte, Yte, Ute, Mte)


class EKF:

    def __init__(self, sm):
        self.sm = sm
        self.m, self.n = sm.m, sm.n
        self.dev = sm.device

    def _jac_h(self, x):
        sm = self.sm
        xs = x.squeeze(-1).squeeze(0)
        H = torch.zeros(self.n, self.m, device=self.dev)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 12] = 1.0; H[2, 14] = 1.0
        H[3, 13] = 1.0; H[3, 15] = 1.0
        def fill(row, i, j):
            dx = xs[i] - xs[j]; dy = xs[i+1] - xs[j+1]
            d = torch.sqrt(dx**2 + dy**2 + 1e-9)
            H[row, i]   =  dx/d; H[row, i+1] =  dy/d
            H[row, j]   = -dx/d; H[row, j+1] = -dy/d
        fill(4, 0, 8)
        fill(5, 8, 16)
        fill(6, 0, 16)
        return H

    def run(self, Y, U, M, x0=None, P0=None):
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

            x = sm.f(x, u, true=False)
            F = sm.F_filter
            P = F @ P @ F.T + sm.Q

            mask = M[k]
            idx = torch.nonzero(mask > 0).squeeze(-1)
            if idx.numel() > 0:
                Hf = self._jac_h(x)[idx]
                yhat = sm.h(x).squeeze(0)[idx]
                ymeas = Y[k][idx]
                Rk = sm.R[idx][:, idx]
                innov = ymeas - yhat
                S = Hf @ P @ Hf.T + Rk
                K = P @ Hf.T @ torch.linalg.inv(S)
                x = x + (K @ innov).reshape(1, self.m, 1)
                A = I - K @ Hf
                P = A @ P @ A.T + K @ Rk @ K.T
            x_hist[k] = x.squeeze(0); P_hist[k] = P
        return x_hist, P_hist


class KalmanNetNN(nn.Module):

    def __init__(self, sm, archi="archi2",
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

    def _build_archi1(self):
        m, n = self.m, self.n
        d_in  = n + m
        self.h_dim = 10 * (m**2 + n**2)
        self.fc_in = nn.Sequential(nn.Linear(d_in, d_in * self.in_mult), nn.ReLU())
        self.gru   = nn.GRU(d_in * self.in_mult, self.h_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(self.h_dim, (m * n) * 4), nn.ReLU(),
            nn.Linear((m * n) * 4, m * n))

    def _build_archi2(self):
        m, n, im, om = self.m, self.n, self.in_mult, self.out_mult
        self.d_hidden_Q     = m * m
        self.d_hidden_Sigma = m * m
        self.d_hidden_S     = n * n
        self.FC5   = nn.Sequential(nn.Linear(m, m * im), nn.ReLU())
        self.GRU_Q = nn.GRU(m * im, self.d_hidden_Q)
        self.FC6   = nn.Sequential(nn.Linear(m, m * im), nn.ReLU())
        self.GRU_Sigma = nn.GRU(self.d_hidden_Q + m * im, self.d_hidden_Sigma)
        self.FC1   = nn.Sequential(nn.Linear(self.d_hidden_Sigma, n*n), nn.ReLU())
        self.FC7   = nn.Sequential(nn.Linear(2*n, 2*n*im), nn.ReLU())
        self.GRU_S = nn.GRU(n*n + 2*n*im, self.d_hidden_S)
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_hidden_S + self.d_hidden_Sigma,
                      (self.d_hidden_S + self.d_hidden_Sigma) * om), nn.ReLU(),
            nn.Linear((self.d_hidden_S + self.d_hidden_Sigma) * om, n * m))
        self.FC3 = nn.Sequential(nn.Linear(self.d_hidden_S + n*m, m*m), nn.ReLU())
        self.FC4 = nn.Sequential(nn.Linear(self.d_hidden_Sigma + m*m,
                                           self.d_hidden_Sigma), nn.ReLU())

    def init_sequence(self, x0, batch):
        self.batch = batch
        x0 = x0.to(self.dev)
        if x0.dim() == 2:
            self.m1x_post = x0.reshape(1, self.m, 1).repeat(batch, 1, 1)
        else:
            self.m1x_post = x0.clone()
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

    def step_prior(self, u):
        self.m1x_prior = self.f(self.m1x_post, u, true=False)
        self.m1y       = self.h(self.m1x_prior)

    def detach_state(self):
        self.m1x_post       = self.m1x_post.detach()
        self.m1x_post_prev  = self.m1x_post_prev.detach()
        self.m1x_prior_prev = self.m1x_prior_prev.detach()
        self.y_prev         = self.y_prev.detach()
        if self.archi == "archi1":
            self.hid = self.hid.detach()
        else:
            self.h_Q     = self.h_Q.detach()
            self.h_Sigma = self.h_Sigma.detach()
            self.h_S     = self.h_S.detach()

    def _features(self, y):
        f2 = (y - self.m1y).squeeze(-1)
        f4 = (self.m1x_post - self.m1x_prior_prev).squeeze(-1)
        f1 = (y - self.y_prev).squeeze(-1)
        f3 = (self.m1x_post - self.m1x_post_prev).squeeze(-1)
        norm = lambda v: F.normalize(v, p=2, dim=1, eps=1e-12)
        return norm(f1), norm(f2), norm(f3), norm(f4)

    def _kgain_archi1(self, f2, f4):
        x = torch.cat([f2, f4], dim=1).unsqueeze(0)
        x = self.fc_in(x)
        out, self.hid = self.gru(x, self.hid)
        kg = self.fc_out(out)
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
        out_FC2 = self.FC2(torch.cat([out_Sigma, out_S], dim=2))
        out_FC3 = self.FC3(torch.cat([out_S, out_FC2], dim=2))
        out_FC4 = self.FC4(torch.cat([out_Sigma, out_FC3], dim=2))
        self.h_Sigma = out_FC4
        return out_FC2.reshape(self.batch, self.m, self.n)

    def step(self, y, u, mask):
        self.step_prior(u)
        f1, f2, f3, f4 = self._features(y)
        if self.archi == "archi1":
            KG = self._kgain_archi1(f2, f4)
        else:
            KG = self._kgain_archi2(f1, f2, f3, f4)
        self.KGain = KG
        dy = (y - self.m1y) * mask.reshape(1, self.n, 1)
        inov = torch.bmm(KG, dy)
        self.m1x_post_prev  = self.m1x_post
        self.m1x_post       = self.m1x_prior + inov
        self.m1x_prior_prev = self.m1x_prior
        self.y_prev = y
        return self.m1x_post

    def forward(self, y, u, mask):
        return self.step(y, u, mask)


def train(sm, model, data_train, data_val, tag="archi2"):
    Xtr, Ytr, Utr, Mtr = data_train
    Xva, Yva, Uva, Mva = data_val
    N, Tp1 = Xtr.shape[0], Xtr.shape[1]
    T = Tp1 - 1

    opt = torch.optim.Adam(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)

    def weighted_mse(xhat, xtrue):
        return ((xhat - xtrue)**2 * sm.loss_w).mean()

    hist_train, hist_val = [], []
    best_val = float('inf')
    best_path = os.path.join(CFG.OUT_DIR, f"knet_{tag}.pt")
    tbptt = getattr(CFG, "TBPTT", 20)

    for epoch in range(CFG.N_EPOCHS):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_skip = 0
        n_batches = N // CFG.N_BATCH
        for bi in range(n_batches):
            idx = perm[bi*CFG.N_BATCH:(bi+1)*CFG.N_BATCH]
            Xb, Yb, Ub, Mb = Xtr[idx], Ytr[idx], Utr[idx], Mtr[idx]
            b = Xb.shape[0]
            model.init_sequence(sm.x0, b)
            opt.zero_grad()
            window_loss = 0.0
            n_win = 0
            for k in range(1, T + 1):
                xhat = model(Yb[:, k], Ub[:, k-1], Mb[0, k])
                window_loss = window_loss + weighted_mse(xhat, Xb[:, k])
                n_win += 1
                if (k % tbptt == 0) or (k == T):
                    wl = window_loss / n_win
                    wl.backward()
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
                    if torch.isfinite(gnorm) and torch.isfinite(wl).all():
                        opt.step()
                    else:
                        n_skip += 1
                    opt.zero_grad()
                    epoch_loss += wl.item() if torch.isfinite(wl).all() else 0.0
                    model.detach_state()
                    window_loss = 0.0; n_win = 0
        if n_skip > 0:
            print(f"    [{tag}] epoch {epoch}: {n_skip} pas sautés (gradient non-fini)")
        hist_train.append(epoch_loss / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            b = Xva.shape[0]
            model.init_sequence(sm.x0, b)
            vloss = 0.0
            for k in range(1, T + 1):
                xhat = model(Yva[:, k], Uva[:, k-1], Mva[0, k])
                vloss = vloss + weighted_mse(xhat, Xva[:, k]).item()
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


def monte_carlo_knet(sm, model, seed=777, n_mc=CFG.N_MC):
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n_mc):
        X, Y, U, Mk = generate_trajectory(sm, rng)
        xh = run_knet(sm, model, Y, U, Mk)
        errs.append((xh - X).squeeze(-1).cpu().numpy())
    errs = np.stack(errs)
    sigma_emp = errs.std(axis=0)
    return sigma_emp


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
        lim = max(np.percentile(np.abs(err_ekf[np.isfinite(err_ekf)]), 99),
                  3*np.nanmax(sig_ekf), 1e-3)
        axs[i].set_ylim(-1.15*lim, 1.15*lim)
        kmax = np.nanmax(np.abs(err_knet[np.isfinite(err_knet)])) if np.isfinite(err_knet).any() else np.inf
        if not np.isfinite(err_knet).all() or kmax > 5*lim:
            axs[i].text(0.5, 0.92,
                        "KNet hors échelle (diverge / non entraîné)",
                        transform=axs[i].transAxes, ha='center', va='top',
                        fontsize=8, color='red',
                        bbox=dict(fc='white', ec='red', alpha=0.7))
    axs[6].set_xlabel("Temps (s)"); axs[7].set_xlabel("Temps (s)")
    fig.suptitle(f"Drone {drone} — erreurs EKF vs KalmanNet ({tag})",
                 fontsize=13, fontweight='bold')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(outdir, f"drone{drone}_{tag}.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def mse_db_curve(sm, model, r_levels_db, n_mc, seed=2024, ekf=None):
    knet_db, ekf_db = [], []
    for r_db in r_levels_db:
        r_scale = 10 ** (-r_db / 20.0)
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
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n_mc):
        X, Y, U, Mk = generate_trajectory(sm, rng)
        xh = run_knet(sm, model, Y, U, Mk)
        errs.append((xh - X).squeeze(-1).cpu().numpy())
    errs = np.stack(errs)
    Pnp = P_ekf.cpu().numpy()
    Tp1, m = errs.shape[1], errs.shape[2]
    nci = np.zeros(Tp1); cover = np.zeros(Tp1)
    for k in range(Tp1):
        Pk = Pnp[k] + 1e-9*np.eye(m)
        Pinv = np.linalg.inv(Pk)
        sig = np.sqrt(np.diag(Pk))
        ek = errs[:, k, :]
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

    print(f">> Device : {CFG.DEVICE} | CUDA dispo : {torch.cuda.is_available()}")
    if CFG.DEVICE.type == "cuda":
        print(f">> GPU : {torch.cuda.get_device_name(0)}")

    print("== Préparation des données ==")
    if CFG.USE_SAVED_DATASET:
        data_train, data_val, (Xte, Yte, Ute, Mte) = load_dataset(sm)
    else:
        data_train = generate_dataset(sm, CFG.N_TRAIN, seed=CFG.SEED,      noise_sweep=CFG.TRAIN_NOISE_SWEEP)
        data_val   = generate_dataset(sm, CFG.N_VAL,   seed=CFG.SEED + 1,  noise_sweep=CFG.TRAIN_NOISE_SWEEP)
        data_test  = generate_dataset(sm, CFG.N_TEST,  seed=CFG.SEED + 99, noise_sweep=False)
        if CFG.TRAIN_NOISE_SWEEP:
            print(f">> Entrainement avec variation de bruit 1/r2 dans {CFG.TRAIN_NOISE_DB} dB")
        save_dataset(data_train, data_val, data_test)
        Xte, Yte, Ute, Mte = (data_test[0][0], data_test[1][0],
                              data_test[2][0], data_test[3][0])
    temps = (np.arange(CFG.T + 1) * sm.dt)

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

        state = torch.load(ckpt, map_location=sm.device)
        model.load_state_dict(state['state_dict'])

        x_knet = run_knet(sm, model, Yte, Ute, Mte)

        sigma_mc = None
        if CFG.MODE_MONTE_CARLO:
            print(f"  Monte-Carlo {CFG.N_MC} runs ({archi})...")
            sigma_mc = monte_carlo_knet(sm, model)

        for d in (1, 2, 3):
            produced.append(plot_drone(sm, d, Xte, x_ekf, x_knet, P_ekf,
                                       temps, archi, sigma_mc=sigma_mc))

        mse_pos = ((x_knet[:, 8:10, 0] - Xte[:, 8:10, 0])**2).mean().item()
        mse_ekf = ((x_ekf[:, 8:10, 0]  - Xte[:, 8:10, 0])**2).mean().item()
        print(f"  MSE pos drone2 : KNet={mse_pos:.4f} | EKF={mse_ekf:.4f}")

        if CFG.PLOT_MSE_DB:
            print(f"  Courbe MSE[dB] ({archi})...")
            r_db, k_db, e_db = mse_db_curve(sm, model, CFG.R_SWEEP_DB,
                                            CFG.N_MC_DB, ekf=ekf)
            produced.append(plot_mse_db(r_db, k_db, e_db, archi, CFG.OUT_DIR))

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


"""
Familles de commandes pour l'etude de generalisation OOD.

Module volontairement AUTONOME : il n'importe rien de KalmanNet_Drones.py
(pas d'import circulaire). Il ne manipule que (T, dt, rng).

Convention : chaque generateur produit un signal d'acceleration 2D (T, 2),
replique a l'identique sur les 3 drones -> u de dimension 6.
La formation reste donc rigide (choix assume : on isole l'axe "regime
dynamique" sans toucher a la geometrie).

Sortie : torch.Tensor de forme (T, 6, 1), float32 -- identique a ce que
retourne build_command_sequence() dans KalmanNet_Drones.py.
"""

import numpy as np
import torch


# ---------------------------------------------------------------- familles
# Entrainement (modele B) : couverture large, structure temporelle variable.
FAMILIES_TRAIN = ("phases3_rand", "ou")

# Test : JAMAIS vues a l'entrainement (leave-one-family-out strict).
FAMILIES_TEST = ("creneaux", "chirp", "virage", "stopgo")

# Reference in-distribution (= build_command_sequence d'origine).
FAMILY_REF = "phases3_ref"


def _tile3(a2):
    """(T, 2) -> (T, 6) : meme commande pour les 3 drones."""
    return np.concatenate([a2, a2, a2], axis=1).astype(np.float32)


# ------------------------------------------------------------ entrainement
def _phases3_ref(T, dt, rng, A=None):
    """Famille d'origine : 3 phases, A ~ U(0.9, 1.1), bornes figees a T/3."""
    A = rng.uniform(0.9, 1.1) if A is None else float(A)
    phi_x = rng.uniform(0, 2 * np.pi)
    phi_y = rng.uniform(0, 2 * np.pi)
    a = np.zeros((T, 2), dtype=np.float32)
    for k in range(T):
        if k < T / 3 or k >= 2 * T / 3:
            a[k] = [A, 0.0]
        else:
            phi_x += 5 * dt
            phi_y += 1 * dt
            a[k] = [A * np.cos(phi_x), A * np.sin(phi_y)]
    return a


def _phases3_rand(T, dt, rng, A=None):
    """3 phases generalisees : amplitude, pulsations, cap et bornes aleatoires.

    Les bornes de phase sont tirees a chaque trajectoire : c'est ce qui
    empeche la GRU d'apprendre une horloge ("la manoeuvre commence au pas 53").
    """
    A = rng.uniform(0.5, 2.5) if A is None else float(A)
    wx = rng.uniform(1.0, 8.0)
    wy = rng.uniform(0.3, 3.0)
    phi_x = rng.uniform(0, 2 * np.pi)
    phi_y = rng.uniform(0, 2 * np.pi)
    cap = rng.uniform(0, 2 * np.pi)          # direction des phases rectilignes
    c, s = np.cos(cap), np.sin(cap)

    f1 = rng.uniform(0.15, 0.45)
    f2 = rng.uniform(f1 + 0.20, 0.90)
    k1, k2 = int(f1 * T), int(f2 * T)

    a = np.zeros((T, 2), dtype=np.float32)
    for k in range(T):
        if k < k1 or k >= k2:
            a[k] = [A * c, A * s]
        else:
            phi_x += wx * dt
            phi_y += wy * dt
            a[k] = [A * np.cos(phi_x), A * np.sin(phi_y)]
    return a


def _ou(T, dt, rng, A=None):
    """Processus d'Ornstein-Uhlenbeck : bruit colore, aucune structure temporelle.

    Couvre un continuum de contenus frequentiels selon theta, sans jamais
    presenter de discontinuite ni de periodicite.
    """
    theta = rng.uniform(0.5, 3.0)
    sig_stat = rng.uniform(0.4, 1.6) if A is None else float(A)
    sigma = sig_stat * np.sqrt(2 * theta)

    a = np.zeros((T, 2), dtype=np.float32)
    cur = rng.normal(0.0, sig_stat, size=2)
    for k in range(T):
        cur = cur - theta * cur * dt + sigma * np.sqrt(dt) * rng.normal(0, 1, size=2)
        a[k] = cur
    return a


# -------------------------------------------------------------------- test
def _creneaux(T, dt, rng, A=None):
    """Bang-bang : paliers constants, transitions discontinues, axes cardinaux.

    Rupture nette avec l'entrainement : la derivee de u est un Dirac.
    """
    a = np.zeros((T, 2), dtype=np.float32)
    k = 0
    while k < T:
        seg = int(rng.integers(10, 41))
        amp = rng.uniform(1.0, 2.5) if A is None else float(A)
        ang = float(rng.choice([0.0, np.pi / 2, np.pi, -np.pi / 2]))
        a[k:k + seg] = [amp * np.cos(ang), amp * np.sin(ang)]
        k += seg
    return a


def _chirp(T, dt, rng, A=None):
    """Chirp lineaire : la frequence balaie une plage jamais vue en continu.

    Teste l'extrapolation frequentielle : l'entrainement ne contient que des
    pulsations fixes par trajectoire.
    """
    amp = rng.uniform(0.8, 2.0) if A is None else float(A)
    f0 = rng.uniform(0.05, 0.20)
    f1 = rng.uniform(0.80, 2.00)
    t = np.arange(T) * dt
    Tt = max(T * dt, 1e-9)
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t ** 2 / Tt)
    a = np.stack([amp * np.cos(phase), amp * np.sin(0.5 * phase)], axis=1)
    return a.astype(np.float32)


def _virage(T, dt, rng, A=None):
    """Virage coordonne : vecteur d'acceleration de module constant en rotation.

    Regime le plus "physique" des quatre, mais absent de l'entrainement :
    les deux composantes sont en quadrature exacte et de meme pulsation.
    """
    amp = rng.uniform(0.8, 2.0) if A is None else float(A)
    Om = float(rng.choice([-1.0, 1.0])) * rng.uniform(0.2, 1.0)
    ph = rng.uniform(0, 2 * np.pi)
    t = np.arange(T) * dt
    a = np.stack([amp * np.cos(Om * t + ph), amp * np.sin(Om * t + ph)], axis=1)
    return a.astype(np.float32)


def _stopgo(T, dt, rng, A=None):
    """Stop-and-go : alternance de plateaux a acceleration nulle et de poussees.

    Cible specifiquement le drone 2, dont le filtre suppose une acceleration
    constante (F_filter, accel_const=True) : les mises a zero brutales sont
    le pire cas pour cette hypothese.
    """
    a = np.zeros((T, 2), dtype=np.float32)
    k = 0
    on = bool(rng.integers(0, 2))
    while k < T:
        seg = int(rng.integers(8, 31))
        if on:
            amp = rng.uniform(1.0, 2.5) if A is None else float(A)
            ang = rng.uniform(0, 2 * np.pi)
            a[k:k + seg] = [amp * np.cos(ang), amp * np.sin(ang)]
        k += seg
        on = not on
    return a


# ------------------------------------------------------------------ facade
_GENERATORS = {
    "phases3_ref":  _phases3_ref,
    "phases3_rand": _phases3_rand,
    "ou":           _ou,
    "creneaux":     _creneaux,
    "chirp":        _chirp,
    "virage":       _virage,
    "stopgo":       _stopgo,
}

LABELS_FR = {
    "phases3_ref":  "3 phases (reference in-distrib.)",
    "phases3_rand": "3 phases randomisees",
    "ou":           "Ornstein-Uhlenbeck",
    "creneaux":     "Creneaux (bang-bang)",
    "chirp":        "Chirp",
    "virage":       "Virage coordonne",
    "stopgo":       "Stop-and-go",
}


def build_command(T, dt, rng, kind="phases3_rand", A=None):
    """Retourne une sequence de commande (T, 6, 1) pour la famille demandee.

    A : si fourni, force l'amplitude (utilise pour le balayage parametrique).
        Sinon elle est tiree dans la plage propre a la famille.
    """
    if kind not in _GENERATORS:
        raise ValueError(f"famille inconnue : {kind!r}. "
                         f"Disponibles : {sorted(_GENERATORS)}")
    a2 = _GENERATORS[kind](T, dt, rng, A=A)
    u = _tile3(a2)
    return torch.tensor(u, dtype=torch.float32).unsqueeze(-1)


def sample_train_family(rng, families=FAMILIES_TRAIN):
    """Tire une famille d'entrainement uniformement."""
    return families[int(rng.integers(len(families)))]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for k in _GENERATORS:
        u = build_command(160, 0.1, rng, kind=k)
        assert u.shape == (160, 6, 1), (k, u.shape)
        assert torch.isfinite(u).all(), k
        print(f"{k:14s} shape={tuple(u.shape)}  "
              f"|a|_max={u[:, :2, 0].abs().max():.2f}")


"""
Evaluation de la generalisation OOD de KalmanNet.

Compare, sur des familles de commandes jamais vues a l'entrainement :
  - modele A : entraine sur la distribution etroite d'origine
  - modele B : entraine avec randomisation de domaine sur u
  - EKF      : reference invariante (ne "generalise" pas, se degrade
               uniquement par mismatch modele)

Metrique centrale : Delta_dB = 10 log10( MSE_KNet / MSE_EKF ).
  Delta_dB < 0  -> KalmanNet bat l'EKF
  Delta_dB > 0  -> KalmanNet est battu

On raisonne en RATIO et jamais en MSE absolue : quand l'amplitude des
commandes augmente, l'EKF du drone 2 se degrade mecaniquement (son
hypothese d'acceleration constante devient plus fausse). Une MSE absolue
laisserait croire a un merite qui vient en fait de la difficulte du cas.

Prerequis : les deux entrainements ont ete lances (cf. PATCH.md).
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from KalmanNet_Drones import (CFG, SystemModel, EKF, KalmanNetNN,
                              generate_trajectory, run_knet)
from ood_commands import build_command, FAMILIES_TEST, FAMILY_REF, LABELS_FR


# ----------------------------------------------------------------- reglages
MODELS = {                       # nom affiche -> dossier de checkpoints
    "A (etroit)":   "./Dataset",
    "B (randomise)": "./Dataset_ood",
}
ARCHIS   = ["archi1", "archi2"]
FAMILIES = [FAMILY_REF] + list(FAMILIES_TEST)

N_MC      = 20                   # trajectoires par famille
SEED_EVAL = 12345

A_SWEEP       = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
SWEEP_FAMILY  = "creneaux"
N_MC_SWEEP    = 12

DIVERGENCE_FACTOR = 100.0        # run divergent si MSE_KNet > 100 * MSE_EKF
OUT_DIR = "./eval_ood"

# Indices d'etat. Drone 2 = le drone critique : ni GPS, ni commande connue
# du filtre (B_filter, estim_d2=False), et hypothese d'acceleration
# constante (F_filter). C'est la que KalmanNet a quelque chose a apporter.
IDX_POS_D2  = slice(8, 10)
IDX_POS_ALL = [0, 1, 8, 9, 16, 17]


# -------------------------------------------------------------------- outils
def load_models(sm):
    """Charge {nom_modele: {archi: KalmanNetNN}} en ignorant les manquants."""
    out = {}
    for name, folder in MODELS.items():
        out[name] = {}
        for archi in ARCHIS:
            ckpt = os.path.join(folder, f"knet_{archi}.pt")
            if not os.path.exists(ckpt):
                print(f"   !! checkpoint absent, ignore : {ckpt}")
                continue
            state = torch.load(ckpt, map_location=sm.device)
            model = KalmanNetNN(sm, archi=state.get("archi", archi))
            model.load_state_dict(state["state_dict"])
            model.eval()
            out[name][archi] = model
        if out[name]:
            print(f"   {name:16s} : {sorted(out[name])}")
    return out


def _mse(xa, xb, idx):
    """MSE sur un sous-ensemble de composantes d'etat."""
    if isinstance(idx, slice):
        d = xa[:, idx, 0] - xb[:, idx, 0]
    else:
        d = xa[:, idx, 0] - xb[:, idx, 0]
    return (d ** 2).mean().item()


def eval_family(sm, ekf, models, kind, n_mc, seed, A=None):
    """Evalue tous les modeles sur n_mc trajectoires d'une famille donnee.

    Les MEMES trajectoires servent a tous les modeles et a l'EKF : la
    comparaison est appariee, ce qui elimine la variance de tirage.
    """
    rng = np.random.default_rng(seed)
    acc = {name: {archi: {"knet_d2": [], "knet_all": [], "div": 0}
                  for archi in models[name]}
           for name in models}
    ekf_d2, ekf_all = [], []

    for _ in range(n_mc):
        u = build_command(CFG.T, sm.dt, rng, kind=kind, A=A)
        X, Y, U, M = generate_trajectory(sm, rng, u_seq=u)

        xe, _ = ekf.run(Y, U, M)
        e_d2  = _mse(xe, X, IDX_POS_D2)
        e_all = _mse(xe, X, IDX_POS_ALL)
        ekf_d2.append(e_d2)
        ekf_all.append(e_all)

        for name in models:
            for archi, model in models[name].items():
                xk = run_knet(sm, model, Y, U, M)
                k_d2  = _mse(xk, X, IDX_POS_D2)
                k_all = _mse(xk, X, IDX_POS_ALL)
                bad = (not np.isfinite(k_d2)) or (k_d2 > DIVERGENCE_FACTOR * e_d2)
                if bad:
                    acc[name][archi]["div"] += 1
                if np.isfinite(k_d2):
                    acc[name][archi]["knet_d2"].append(k_d2)
                    acc[name][archi]["knet_all"].append(k_all)

    res = {"ekf_d2": float(np.mean(ekf_d2)), "ekf_all": float(np.mean(ekf_all)),
           "models": {}}
    for name in models:
        res["models"][name] = {}
        for archi in models[name]:
            a = acc[name][archi]
            if not a["knet_d2"]:
                res["models"][name][archi] = None
                continue
            m_d2  = float(np.mean(a["knet_d2"]))
            m_all = float(np.mean(a["knet_all"]))
            res["models"][name][archi] = {
                "mse_d2":   m_d2,
                "mse_all":  m_all,
                "delta_d2":  10 * np.log10(m_d2 / res["ekf_d2"]),
                "delta_all": 10 * np.log10(m_all / res["ekf_all"]),
                "div_rate":  a["div"] / n_mc,
            }
    return res


# ------------------------------------------------------------------ figures
def plot_bars(results, archi, outdir):
    """Delta_dB par famille, un groupe de barres par modele."""
    names = [n for n in MODELS if any(
        results[f]["models"].get(n, {}).get(archi) for f in FAMILIES)]
    if not names:
        return None

    fams = FAMILIES
    xs = np.arange(len(fams))
    width = 0.8 / max(len(names), 1)
    colors = ["#c44e52", "#4c72b0", "#55a868"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, name in enumerate(names):
        vals = []
        for f in fams:
            r = results[f]["models"].get(name, {}).get(archi)
            vals.append(r["delta_d2"] if r else np.nan)
        ax.bar(xs + i * width - 0.4 + width / 2, vals, width,
               label=f"KNet {name}", color=colors[i % len(colors)])

    ax.axhline(0, color="k", lw=1.2)
    ax.text(len(fams) - 0.45, 0.25, "EKF (reference)", fontsize=9,
            ha="right", va="bottom", style="italic")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS_FR[f].replace(" (", "\n(") for f in fams],
                       fontsize=9)
    ax.set_ylabel(r"$\Delta_{dB} = 10\log_{10}(MSE_{KNet}/MSE_{EKF})$")
    ax.set_title(f"Generalisation OOD — position drone 2 — {archi}\n"
                 "en dessous de 0 : KalmanNet bat l'EKF", fontsize=12)
    ax.grid(True, axis="y", ls=":", alpha=.7)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(outdir, f"ood_bars_{archi}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_sweep(sweep, archi, outdir):
    """Delta_dB en fonction de l'amplitude : ou se situe le decrochage."""
    names = [n for n in MODELS
             if any(s["models"].get(n, {}).get(archi) for s in sweep.values())]
    if not names:
        return None

    colors = ["#c44e52", "#4c72b0", "#55a868"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, name in enumerate(names):
        vals = [sweep[A]["models"].get(name, {}).get(archi)["delta_d2"]
                if sweep[A]["models"].get(name, {}).get(archi) else np.nan
                for A in A_SWEEP]
        ax.plot(A_SWEEP, vals, "o-", lw=2, label=f"KNet {name}",
                color=colors[i % len(colors)])

    ax.axhline(0, color="k", lw=1.2, label="EKF (reference)")
    ax.axvspan(0.9, 1.1, color="grey", alpha=.20)
    ax.text(1.0, ax.get_ylim()[1], " plage vue\n par le modele A",
            fontsize=8, ha="center", va="top", color="dimgrey")
    ax.set_xlabel("Amplitude de commande A")
    ax.set_ylabel(r"$\Delta_{dB}$")
    ax.set_title(f"Extrapolation en amplitude — famille "
                 f"{LABELS_FR[SWEEP_FAMILY]} — {archi}", fontsize=12)
    ax.grid(True, ls=":", alpha=.7)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(outdir, f"ood_sweep_{archi}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_timeseries(sm, ekf, models, archi, outdir, kind="stopgo", seed=999):
    """Erreur de position drone 2 au cours du temps sur une trajectoire."""
    rng = np.random.default_rng(seed)
    u = build_command(CFG.T, sm.dt, rng, kind=kind)
    X, Y, U, M = generate_trajectory(sm, rng, u_seq=u)
    temps = np.arange(CFG.T + 1) * sm.dt

    xe, _ = ekf.run(Y, U, M)
    err_e = np.linalg.norm((xe[:, 8:10, 0] - X[:, 8:10, 0]).cpu().numpy(), axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(temps, err_e, color="green", lw=1.6, label="EKF")
    colors = ["#c44e52", "#4c72b0", "#55a868"]
    for i, (name, per_archi) in enumerate(models.items()):
        if archi not in per_archi:
            continue
        xk = run_knet(sm, per_archi[archi], Y, U, M)
        err_k = np.linalg.norm((xk[:, 8:10, 0] - X[:, 8:10, 0]).cpu().numpy(), axis=1)
        ax.plot(temps, err_k, lw=1.6, color=colors[i % len(colors)],
                label=f"KNet {name}")

    ax.set_yscale("log")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Erreur de position drone 2 (m)")
    ax.set_title(f"Reponse temporelle — {LABELS_FR[kind]} — {archi}", fontsize=12)
    ax.grid(True, ls=":", alpha=.7)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(outdir, f"ood_timeseries_{archi}_{kind}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# --------------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(CFG.SEED)
    np.random.seed(CFG.SEED)

    # Securite : l'evaluation impose ses propres commandes via u_seq, mais on
    # neutralise le flag pour qu'aucun appel indirect ne rebascule dessus.
    CFG.TRAIN_CMD_RANDOMIZE = False

    sm = SystemModel()
    ekf = EKF(sm)

    print("== Chargement des modeles ==")
    models = load_models(sm)
    models = {k: v for k, v in models.items() if v}
    if not models:
        print("!! Aucun checkpoint trouve. Lance les deux entrainements d'abord.")
        return

    print(f"\n== Evaluation par famille ({N_MC} trajectoires each) ==")
    results = {}
    for fam in FAMILIES:
        results[fam] = eval_family(sm, ekf, models, fam, N_MC, SEED_EVAL)
        print(f"\n  {LABELS_FR[fam]}")
        print(f"     EKF   MSE pos d2 = {results[fam]['ekf_d2']:.4f}")
        for name in models:
            for archi in models[name]:
                r = results[fam]["models"][name][archi]
                if r is None:
                    print(f"     {name:14s} {archi} : diverge integralement")
                    continue
                flag = "  <-- battu par l'EKF" if r["delta_d2"] > 0 else ""
                print(f"     {name:14s} {archi} : MSE={r['mse_d2']:.4f}  "
                      f"Delta={r['delta_d2']:+6.2f} dB  "
                      f"div={100*r['div_rate']:.0f}%{flag}")

    print(f"\n== Balayage en amplitude ({LABELS_FR[SWEEP_FAMILY]}) ==")
    sweep = {}
    for A in A_SWEEP:
        sweep[A] = eval_family(sm, ekf, models, SWEEP_FAMILY,
                               N_MC_SWEEP, SEED_EVAL + 7, A=A)
        line = f"  A={A:.1f} | EKF={sweep[A]['ekf_d2']:8.3f}"
        for name in models:
            for archi in models[name]:
                r = sweep[A]["models"][name][archi]
                line += f" | {name[0]}-{archi[-1]}={r['delta_d2']:+6.2f}dB" if r else " | n/a"
        print(line)

    print("\n== Figures ==")
    produced = []
    for archi in ARCHIS:
        if not any(archi in v for v in models.values()):
            continue
        for fn, args in ((plot_bars, (results, archi, OUT_DIR)),
                         (plot_sweep, (sweep, archi, OUT_DIR))):
            p = fn(*args)
            if p:
                produced.append(p)
        for kind in ("stopgo", "creneaux"):
            p = plot_timeseries(sm, ekf, models, archi, OUT_DIR, kind=kind)
            if p:
                produced.append(p)

    summary = os.path.join(OUT_DIR, "resultats_ood.json")
    with open(summary, "w") as fh:
        json.dump({"familles": results,
                   "balayage": {str(k): v for k, v in sweep.items()}},
                  fh, indent=2)
    produced.append(summary)

    for p in produced:
        print("  ", p)


if __name__ == "__main__":
    main()
