def load_dataset(sm, path=None):
    """
    Charge le dataset.npz produit par make_dataset.py et reconstruit les
    tenseurs (train, val, test) selon le split enregistré.
    """
    if path is None:
        path = CFG.DATASET_PATH or os.path.join(CFG.OUT_DIR, "dataset.npz")
    d = np.load(path)
    X, Y, U, M = d["X"], d["Y"], d["U"], d["M"]
    itr, iva, ite = d["idx_train"], d["idx_val"], d["idx_test"]

    def pack(idx):
        return (torch.tensor(X[idx], dtype=torch.float32, device=sm.device),
                torch.tensor(Y[idx], dtype=torch.float32, device=sm.device),
                torch.tensor(U[idx], dtype=torch.float32, device=sm.device),
                torch.tensor(M[idx], dtype=torch.float32, device=sm.device))

    data_train = pack(itr)
    data_val   = pack(iva)
    j = ite[0]
    Xte = torch.tensor(X[j], dtype=torch.float32, device=sm.device)
    Yte = torch.tensor(Y[j], dtype=torch.float32, device=sm.device)
    Ute = torch.tensor(U[j], dtype=torch.float32, device=sm.device)
    Mte = torch.tensor(M[j], dtype=torch.float32, device=sm.device)
    print(f">> Dataset chargé : {path}")
    print(f"   train={len(itr)}  val={len(iva)}  test={len(ite)}")
    return data_train, data_val, (Xte, Yte, Ute, Mte)


def build_command_ood(T, dt, rng, kind="3phases"):
    """
    Commandes OOD : régimes dynamiques NON vus à l'entraînement.
    """
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
