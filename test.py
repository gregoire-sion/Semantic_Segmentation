import os
import numpy as np
import torch

from KalmanNet_Drones import (
    CFG, SystemModel, EKF, KalmanNetNN,
    build_command_sequence, run_knet, plot_drone,
)


ARCHIS       = ["archi1", "archi2"]
CKPT_DIR     = "Entrainement_presentationmistage"
OUT_DIR      = "Entrainement_presentationmistage/test_offset"
SEED_TEST    = CFG.SEED + 99
OFFSET_SCALE = 1.0


def gen_trajectory_offset(sm, rng, scale=1.0, r_scale=1.0):
    T, m, n = CFG.T, sm.m, sm.n
    dev = sm.device
    U = build_command_sequence(T, sm.dt, rng).to(dev)
    sqrtR = torch.linalg.cholesky(sm.R_gen) * r_scale

    X = torch.zeros(T + 1, m, 1, device=dev)
    Y = torch.zeros(T + 1, n, 1, device=dev)
    M = torch.zeros(T + 1, n, device=dev)

    x = sm.x0.clone()
    L = torch.linalg.cholesky(sm.P0)
    xi = torch.tensor(rng.normal(0, 1, size=m), dtype=torch.float32, device=dev).reshape(m, 1)
    offset_mask = torch.ones(m, 1, device=dev)
    for b in (0, 16):
        offset_mask[b+4] = offset_mask[b+5] = 0.0
        offset_mask[b+6] = offset_mask[b+7] = 0.0
    x = x + scale * (L @ xi) * offset_mask
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(CFG.SEED); np.random.seed(CFG.SEED)
    sm = SystemModel()
    ekf = EKF(sm)

    rng = np.random.default_rng(SEED_TEST)
    Xte, Yte, Ute, Mte = gen_trajectory_offset(sm, rng, scale=OFFSET_SCALE)
    temps = np.arange(CFG.T + 1) * sm.dt

    x_ekf, P_ekf = ekf.run(Yte, Ute, Mte)

    d0 = (Xte[0] - sm.x0).squeeze(-1).cpu().numpy()
    print(">> Decalage initial verite vs x0 nominal (scale=%.2f)" % OFFSET_SCALE)
    for name, b in (("drone1", 0), ("drone2", 8), ("drone3", 16)):
        print(f"   {name}: dpos=({d0[b]:+.2f},{d0[b+1]:+.2f}) "
              f"dvit=({d0[b+2]:+.2f},{d0[b+3]:+.2f}) "
              f"dacc=({d0[b+4]:+.2f},{d0[b+5]:+.2f}) "
              f"dbias=({d0[b+6]:+.2f},{d0[b+7]:+.2f})")

    produced = []
    for archi in ARCHIS:
        ckpt = os.path.join(CKPT_DIR, f"knet_{archi}.pt")
        if not os.path.exists(ckpt):
            print(f"!! checkpoint introuvable : {ckpt}")
            continue
        state = torch.load(ckpt, map_location=sm.device)
        model = KalmanNetNN(sm, archi=state.get('archi', archi))
        model.load_state_dict(state['state_dict'])

        x_knet = run_knet(sm, model, Yte, Ute, Mte)

        tag = f"{archi}_offset"
        for d in (1, 2, 3):
            produced.append(plot_drone(sm, d, Xte, x_ekf, x_knet, P_ekf,
                                       temps, tag, outdir=OUT_DIR))

        mse_pos = ((x_knet[:, 8:10, 0] - Xte[:, 8:10, 0])**2).mean().item()
        mse_ekf = ((x_ekf[:, 8:10, 0]  - Xte[:, 8:10, 0])**2).mean().item()
        print(f"  [{archi}] MSE pos drone2 : KNet={mse_pos:.4f} | EKF={mse_ekf:.4f}")

    print("\n== Figures ==")
    for p in produced:
        print("  ", p)


if __name__ == "__main__":
    main()