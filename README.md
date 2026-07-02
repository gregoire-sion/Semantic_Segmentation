import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, eigvalsh
from scipy.linalg import block_diag
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

t_max      = 16.0
dt         = 0.1
dt_capteur = 0.5
dt_imu     = 0.1
N_DRONES   = 3
N_VAR      = 8
N          = N_DRONES * N_VAR
n_steps    = int(round(t_max / dt))
ratio_imu  = int(round(dt_imu / dt))
ratio_gps  = int(round(dt_capteur / dt))

sigma_w_accel = 5e-2
sigma_w_autre = 1e-6
sigma_gps  = 0.5
sigma_acc  = 0.01
sigma_d    = 0.5
sigma_R_gps = 0.5
sigma_R_acc = 0.1
sigma_R_d   = 0.5
sigma_P_x, sigma_P_v, sigma_P_a, sigma_P_b = 2.0, 0.5, 0.5, 1.0

def Fmat(accel_constante):
    a = 1.0 if accel_constante else 0.0
    return np.array([
        [1, 0, dt, 0, 0.5*dt**2, 0,         0, 0],
        [0, 1, 0,  dt, 0,        0.5*dt**2, 0, 0],
        [0, 0, 1,  0,  dt,       0,          0, 0],
        [0, 0, 0,  1,  0,        dt,         0, 0],
        [0, 0, 0,  0,  a,        0,          0, 0],
        [0, 0, 0,  0,  0,        a,          0, 0],
        [0, 0, 0,  0,  0,        0,          1, 0],
        [0, 0, 0,  0,  0,        0,          0, 1]], dtype=float)

def Bmat_global(cols):
    M = np.zeros((8, 6)); M[4, cols[0]] = 1.0; M[5, cols[1]] = 1.0
    return M

def Bmat_local():
    M = np.zeros((8, 2)); M[4, 0] = 1.0; M[5, 1] = 1.0
    return M

F_vrai = block_diag(Fmat(False), Fmat(False), Fmat(False))
B_vrai = np.concatenate((Bmat_global([0,1]), Bmat_global([2,3]), Bmat_global([4,5])), axis=0)

F_loc  = {1: Fmat(False), 2: Fmat(True),  3: Fmat(False)}
B_loc  = {1: Bmat_local(), 2: np.zeros((8,2)), 3: Bmat_local()}

w_sigma = np.full(N, sigma_w_autre)
for b in (0, 8, 16):
    w_sigma[b+4] = w_sigma[b+5] = sigma_w_accel

X_vrai_init = np.concatenate(([0, 10, 1, 0, 0, 0,  0,    0],
                               [10, 0, 1, 0, 0, 0,  0.5, -0.2],
                               [0, -10, 1, 0, 0, 0,  0,    0])).astype(float)

I8 = np.eye(8)

def make_Q_local(drone_id):
    Q = np.eye(8) * 1e-3
    Q[4,4] = Q[5,5] = 0.5**2
    Q[6,6] = Q[7,7] = 1e-9**2
    if drone_id == 2:
        Q[2,2] = Q[3,3] = 0.1**2
        Q[4,4] = Q[5,5] = 1.0
    return Q

def make_P_local(drone_id=None):
    P = np.eye(8)
    P[0,0] = P[1,1] = sigma_P_x**2
    P[2,2] = P[3,3] = sigma_P_v**2
    P[4,4] = P[5,5] = sigma_P_a**2
    P[6,6] = P[7,7] = sigma_P_b**2
    if drone_id == 2:
        P[2,2] = P[3,3] = sigma_P_v**2 + 2
    return P

class MethodeBloc2x2:
    nom   = "M0 - Bloc 2x2 exact"
    label = "M0"
    color = "black"
    ls    = "-"
    n_floats_cov = 3
    @staticmethod
    def build_paquet(drone):
        return {"pos": drone.x[0:2].copy(), "data": drone.P[0:2, 0:2].copy()}
    @staticmethod
    def var_voisin(paquet, Hj):
        return float(Hj @ paquet["data"] @ Hj.T)

class MethodeIsotrope:
    nom   = "M1 - Variance isotrope (trace/2)"
    label = "M1"
    color = "royalblue"
    ls    = "--"
    n_floats_cov = 1
    @staticmethod
    def build_paquet(drone):
        Ppos = drone.P[0:2, 0:2]
        return {"pos": drone.x[0:2].copy(), "data": np.trace(Ppos) / 2.0}
    @staticmethod
    def var_voisin(paquet, Hj):
        return float(paquet["data"])

class MethodeProjetee:
    nom   = "M2 - Variance projetee (Hj.P.Hj)"
    label = "M2"
    color = "darkorange"
    ls    = "-."
    n_floats_cov = 1
    @staticmethod
    def build_paquet(drone):
        return {"pos": drone.x[0:2].copy(), "data": drone.P[0:2, 0:2].copy()}
    @staticmethod
    def var_voisin(paquet, Hj):
        return float(Hj @ paquet["data"] @ Hj.T)

class MethodeMax:
    nom   = "M3 - Variance maximale (lmax)"
    label = "M3"
    color = "crimson"
    ls    = ":"
    n_floats_cov = 1
    @staticmethod
    def build_paquet(drone):
        Ppos = drone.P[0:2, 0:2]
        return {"pos": drone.x[0:2].copy(), "data": float(eigvalsh(Ppos).max())}
    @staticmethod
    def var_voisin(paquet, Hj):
        return float(paquet["data"])

METHODES = [MethodeBloc2x2, MethodeIsotrope, MethodeProjetee, MethodeMax]

def cout_floats(methode):
    return 3 * (2 + methode.n_floats_cov)

class DroneDistribue:
    def __init__(self, drone_id, x_init, P_init, Q_local):
        self.id = drone_id
        self.x = x_init.copy()
        self.P = P_init.copy()
        self.Q = Q_local.copy()
        self.F = F_loc[drone_id]
        self.B = B_loc[drone_id]

    def predict(self, u):
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_local(self, mes, H, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ inv(S)
        innov = mes - H @ self.x
        self.x = self.x + K @ innov
        A = I8 - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T

    def update_CI_batch(self, liens, R_scalaire, methode, diagnostics=None):
        if not liens:
            return
        rows_Hi, innovs, R_diag = [], [], []
        for d_mesure, paquet in liens:
            xi, yi = self.x[0], self.x[1]
            xj, yj = paquet["pos"]
            d_pred = np.hypot(xi - xj, yi - yj)
            if d_pred < 1e-4:
                d_pred = 1e-4
            Hi      = np.zeros(8)
            Hi[0]   = (xi - xj) / d_pred
            Hi[1]   = (yi - yj) / d_pred
            Hj      = np.array([-(xi - xj) / d_pred, -(yi - yj) / d_pred])
            var_j = methode.var_voisin(paquet, Hj)
            if diagnostics is not None and "Ppos_exact" in paquet:
                angle = np.degrees(np.arctan2(Hj[1], Hj[0])) % 360.0
                var_exacte = float(Hj @ paquet["Ppos_exact"] @ Hj.T)
                diagnostics.append((angle, var_j, var_exacte))
            rows_Hi.append(Hi)
            innovs.append(d_mesure - d_pred)
            R_diag.append(R_scalaire + var_j)
        H = np.vstack(rows_Hi)
        innov = np.array(innovs)
        Rv = np.diag(R_diag)
        def cout_CI(omega):
            Pi = self.P / omega
            S = H @ Pi @ H.T + Rv / (1.0 - omega)
            K = Pi @ H.T @ inv(S)
            return np.trace(Pi - K @ H @ Pi)
        sol = minimize_scalar(cout_CI, bounds=(0.01, 0.99), method='bounded')
        omega = sol.x
        Pi = self.P / omega
        S = H @ Pi @ H.T + Rv / (1.0 - omega)
        K = Pi @ H.T @ inv(S)
        self.x = self.x + K @ innov
        self.P = Pi - K @ H @ Pi

def commande_vraie(step, phi):
    phi[0] += 5 * dt
    phi[1] += 1 * dt
    u = np.array([np.cos(phi[0]), np.sin(phi[1]),
                  np.cos(phi[0]), np.sin(phi[1]),
                  np.cos(phi[0]), np.sin(phi[1])])
    if step < n_steps / 3:
        u = np.array([1., 0., 1., 0., 1., 0.])
    elif step < 2 * n_steps / 3:
        phi[0] += 5 * dt
        phi[1] += 1 * dt
        u = np.array([np.cos(phi[0]), np.sin(phi[1]),
                      np.cos(phi[0]), np.sin(phi[1]),
                      np.cos(phi[0]), np.sin(phi[1])])
    else:
        u = np.array([1., 0., 1., 0., 1., 0.])
    return u

def run_une_methode(methode, seed, compenser_biais=True,
                    use_gps=True, use_imu=True, use_distances=True):
    np.random.seed(seed)
    X_vrai = X_vrai_init.copy()

    erreur_init = np.zeros(N)
    for b in (0, 8, 16):
        erreur_init[b:b+2]   = np.random.normal(0, sigma_P_x, size=2)
        erreur_init[b+2:b+4] = np.random.normal(0, sigma_P_v, size=2)
        erreur_init[b+4:b+6] = np.random.normal(0, sigma_P_a, size=2)
        erreur_init[b+6:b+8] = np.random.normal(0, sigma_P_b, size=2)
    erreur_init[6:8]   = [0, 0]
    erreur_init[22:24] = [0, 0]
    X_est0 = X_vrai + erreur_init

    drones = {
        i: DroneDistribue(i, X_est0[(i-1)*8:i*8], make_P_local(i), make_Q_local(i))
        for i in (1, 2, 3)
    }

    if not compenser_biais:
        for i, base in ((1,0),(2,8),(3,16)):
            drones[i].x[6:8] = 0.0
            drones[i].P[6,6] = drones[i].P[7,7] = 1e-8
            drones[i].Q[6,6] = drones[i].Q[7,7] = 1e-8

    err_pos  = np.zeros((n_steps+1, N_DRONES, 2))
    P_pos_tr = np.zeros((n_steps+1, N_DRONES))
    P_pos_m  = np.zeros((n_steps+1, N_DRONES, 2, 2))
    err8     = np.zeros((n_steps+1, N_DRONES, 8))
    Pdiag8   = np.zeros((n_steps+1, N_DRONES, 8))
    diag_var = []

    phi = [0.0, 0.0]

    for k in range(1, n_steps+1):
        step = k - 1
        u_vrai = commande_vraie(step, phi)

        err_cmd = np.random.normal(0, 0.1, size=6); err_cmd[0:2] = 0.0
        u_kalman = u_vrai + err_cmd

        X_vrai = F_vrai @ X_vrai + B_vrai @ u_vrai + np.random.normal(0, 1, N) * w_sigma

        drones[1].predict(u_kalman[0:2])
        drones[2].predict(np.zeros(2))
        drones[3].predict(u_kalman[4:6])

        if use_imu and step % ratio_imu == 0:
            mes = np.array([X_vrai[12] + X_vrai[14] + np.random.normal(0, sigma_acc),
                            X_vrai[13] + X_vrai[15] + np.random.normal(0, sigma_acc)])
            H = np.zeros((2, 8)); H[0,4]=H[0,6]=1.0; H[1,5]=H[1,7]=1.0
            drones[2].update_local(mes, H, np.diag([sigma_R_acc**2]*2))

        if step % ratio_gps == 0:
            if use_gps:
                z = np.array([X_vrai[0] + np.random.normal(0, sigma_gps),
                              X_vrai[1] + np.random.normal(0, sigma_gps)])
                H = np.zeros((2, 8)); H[0,0]=1.0; H[1,1]=1.0
                drones[1].update_local(z, H, np.diag([sigma_R_gps**2]*2))

            if use_distances:
                d12v = np.hypot(X_vrai[0]-X_vrai[8],  X_vrai[1]-X_vrai[9])
                d23v = np.hypot(X_vrai[8]-X_vrai[16], X_vrai[9]-X_vrai[17])
                d13v = np.hypot(X_vrai[0]-X_vrai[16], X_vrai[1]-X_vrai[17])
                z12  = d12v + np.random.normal(0, sigma_d)
                z23  = d23v + np.random.normal(0, sigma_d)
                z13  = d13v + np.random.normal(0, sigma_d)

                paquets = {i: methode.build_paquet(drones[i]) for i in (1,2,3)}
                for i in (1, 2, 3):
                    paquets[i]["Ppos_exact"] = drones[i].P[0:2, 0:2].copy()

                drones[1].update_CI_batch([(z12, paquets[2]), (z13, paquets[3])],
                                           sigma_R_d**2, methode, diagnostics=diag_var)
                drones[2].update_CI_batch([(z12, paquets[1]), (z23, paquets[3])],
                                           sigma_R_d**2, methode, diagnostics=diag_var)
                drones[3].update_CI_batch([(z23, paquets[2]), (z13, paquets[1])],
                                           sigma_R_d**2, methode, diagnostics=diag_var)

        for j, (d, b) in enumerate([(1,0),(2,8),(3,16)]):
            err_pos[k, j]   = drones[d].x[0:2] - X_vrai[b:b+2]
            P_pos_tr[k, j]  = drones[d].P[0,0] + drones[d].P[1,1]
            P_pos_m[k, j]   = drones[d].P[0:2, 0:2]
            err8[k, j]      = drones[d].x[0:8] - X_vrai[b:b+8]
            Pdiag8[k, j]    = np.diag(drones[d].P)[0:8]

    return err_pos, P_pos_tr, P_pos_m, np.array(diag_var) if diag_var else np.empty((0,3)), err8, Pdiag8

SEUIL_3SIGMA = chi2.ppf(0.997, df=2)
T_CONV = 4.0
IDX_CONV = int(round(T_CONV / dt))

def calcule_metriques(methode, N_MC=30, seeds=None, cfg=None):
    if seeds is None:
        seeds = range(N_MC)
    if cfg is None:
        cfg = {}

    all_mse    = np.zeros((N_MC, N_DRONES))
    all_nci    = np.zeros((N_MC, N_DRONES))
    all_trace  = np.zeros((N_MC, n_steps+1, N_DRONES))
    all_rmse_t = np.zeros((N_MC, n_steps+1, N_DRONES))
    nci_min    = np.full(N_DRONES, np.inf)
    n_dans_3s  = np.zeros(N_DRONES)
    n_total    = np.zeros(N_DRONES)
    diag_all   = []
    err_hist = np.zeros((N_MC, n_steps+1, N_DRONES, 8))
    P_hist   = np.zeros((N_MC, n_steps+1, N_DRONES, 8))

    for run, seed in enumerate(seeds):
        err_pos, P_pos_tr, P_pos_m, diag_var, err8, Pdiag8 = run_une_methode(
            methode, seed, **cfg)
        if diag_var.size:
            diag_all.append(diag_var)
        err_hist[run] = err8
        P_hist[run]   = Pdiag8

        for j in range(N_DRONES):
            all_mse[run, j] = np.mean(np.sum(err_pos[IDX_CONV:, j, :]**2, axis=1))
            all_rmse_t[run, :, j] = np.sqrt(np.sum(err_pos[:, j, :]**2, axis=1))
            nci_vals = []
            for k in range(1, n_steps+1):
                e   = err_pos[k, j]
                Pij = P_pos_m[k, j]
                try:
                    d2 = float(e @ inv(Pij) @ e)
                except Exception:
                    continue
                nci_vals.append(d2)
                nci_min[j] = min(nci_min[j], d2)
                n_total[j]   += 1
                if d2 <= SEUIL_3SIGMA:
                    n_dans_3s[j] += 1
            all_nci[run, j] = np.mean(nci_vals) if nci_vals else np.nan
            all_trace[run, :, j] = P_pos_tr[:, j]

    trace_mean = np.mean(all_trace, axis=0)
    trace_std  = np.std(all_trace,  axis=0)
    rmse_t     = np.mean(all_rmse_t, axis=0)
    couverture = n_dans_3s / np.maximum(n_total, 1)
    diag_var = np.vstack(diag_all) if diag_all else np.empty((0, 3))
    if diag_var.size:
        ecart = diag_var[:, 1] - diag_var[:, 2]
        var_err_var = float(np.var(ecart))
    else:
        var_err_var = 0.0

    return {
        "mse": all_mse, "rmse_t": rmse_t, "nci": all_nci, "nci_min": nci_min,
        "couverture": couverture, "trace_mean": trace_mean, "trace_std": trace_std,
        "diag_var": diag_var, "var_err_var": var_err_var,
        "err_hist": err_hist, "P_hist": P_hist,
    }

CONFIGS = [
    ("Scenario D", dict(compenser_biais=True,  use_gps=True, use_imu=True, use_distances=True)),
    ("Scenario C", dict(compenser_biais=False, use_gps=True, use_imu=True, use_distances=True)),
    ("Scenario A", dict(compenser_biais=False, use_gps=True, use_imu=True, use_distances=False)),
    ("Scenario B", dict(compenser_biais=True,  use_gps=True, use_imu=True, use_distances=False)),
]

if __name__ == "__main__":
    N_MC  = 40
    seeds = list(range(N_MC))
    temps = np.arange(n_steps+1) * dt

    labels_drone = ["Drone 1 (GPS)", "Drone 2 (IMU)", "Drone 3 (sans capteur)"]

    for nom_cfg, cfg in CONFIGS:
        print("\n" + "="*74)
        print(f"  {nom_cfg}  |  cfg = {cfg}")
        print("="*74)
        resultats = {}
        for m in METHODES:
            R = calcule_metriques(m, N_MC=N_MC, seeds=seeds, cfg=cfg)
            R["methode"] = m
            R["cout"] = cout_floats(m)
            resultats[m.label] = R

        print(f"{'Meth':<6}{'Cout':>6}{'MSE':>10}{'NCI':>8}{'Couv%':>8}")
        for label, r in resultats.items():
            print(f"{r['methode'].label:<6}{r['cout']:>6d}"
                  f"{r['mse'].mean():>10.3f}{r['nci'].mean():>8.2f}"
                  f"{r['couverture'].mean()*100:>7.1f}")

        print(f"\n  MSE position par drone (t >= {T_CONV}s)")
        print(f"{'Meth':<6}" + "".join(f"{d:>24}" for d in labels_drone))
        for label, r in resultats.items():
            mpd = r["mse"].mean(axis=0)
            print(f"{r['methode'].label:<6}" + "".join(f"{v:>24.3f}" for v in mpd))
