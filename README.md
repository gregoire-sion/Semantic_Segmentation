
"""
Train.py
========
Entrainement de KalmanNet (train + VALIDATION).

Repartition des donnees :
  - train      : mise a jour des poids (backprop)
  - validation : surveillance + selection du meilleur checkpoint
  - test       : PAS ICI -> Simulation.py (etancheite du jeu de test)

Garde-fous anti-NaN integres :
  - lr abaisse a 1e-4 (au lieu de 1e-3)         [suspect NaN #2]
  - clipping de gradient (clip_grad_norm_)
  - model.eval() + torch.no_grad() en validation [evite l'explosion memoire]
  - detection de NaN dans la loss -> arret propre
"""

import torch
import torch.nn as nn

import SystemModel as SM
from KalmanNet import KalmanNet


# =========================================================================
# HYPERPARAMETRES
# =========================================================================
CONFIG = {
    "lr": 1e-4,              # abaisse pour stabiliser (suspect NaN #2)
    "weight_decay": 1e-5,    # regularisation l2 (gamma du papier)
    "grad_clip": 1.0,        # norme max du gradient
    "n_epochs": 100,
    "T_train": 100,          # trajectoires courtes pour l'entrainement (BPTT tronque V2)
    "n_train": 50,           # nb de trajectoires d'entrainement
    "n_val": 10,             # nb de trajectoires de validation
    "batch_size": 10,
    "gru_mult": 2,           # commencer petit, remonter une fois stable
    "q2": 1e-3,
    "r2": 1e-2,
    "save_path": "knet_best.pt",
    "seed": 42,
}


# =========================================================================
# GENERATION DES JEUX TRAIN / VALIDATION
# =========================================================================
def make_dataset(n_traj, T, q2, r2, seed):
    """Genere un jeu de n_traj trajectoires. Renvoie X [n,M,T+1], Y [n,N,T]."""
    X, Y = SM.generate(T=T, batch=n_traj, q2=q2, r2=r2, seed=seed)
    return X, Y


# =========================================================================
# BOUCLE D'ENTRAINEMENT
# =========================================================================
def train(cfg=CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    torch.manual_seed(cfg["seed"])

    # --- Donnees ---
    X_tr, Y_tr = make_dataset(cfg["n_train"], cfg["T_train"],
                              cfg["q2"], cfg["r2"], seed=cfg["seed"])
    X_val, Y_val = make_dataset(cfg["n_val"], cfg["T_train"],
                                cfg["q2"], cfg["r2"], seed=cfg["seed"] + 1)
    X_tr, Y_tr = X_tr.to(device), Y_tr.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)

    # On compare X_hat[:, :, t] a l'etat ground-truth X[:, :, t+1]
    # (l'estimation a l'instant t correspond a l'etat apres f).
    Xtgt_tr = X_tr[:, :, 1:]      # [n, M, T]
    Xtgt_val = X_val[:, :, 1:]

    # --- Modele ---
    net = KalmanNet(gru_mult=cfg["gru_mult"]).to(device)
    print(f"Nb parametres : {sum(p.numel() for p in net.parameters()):,}")

    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    mse = nn.MSELoss()

    n_train = X_tr.shape[0]
    best_val = float("inf")

    for epoch in range(cfg["n_epochs"]):
        # ---------------- TRAIN ----------------
        net.train()
        # La sequence de commandes doit etre injectee AVANT le forward,
        # identique a celle du data-gen.
        SM.set_command_sequence(SM.build_command_sequence(cfg["T_train"]))

        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            Yb, Xb = Y_tr[idx], Xtgt_tr[idx]

            opt.zero_grad()
            X_hat = net(Yb)
            loss = mse(X_hat, Xb)

            if not torch.isfinite(loss):
                print(f"[epoch {epoch}] Loss non finie -> arret. "
                      f"Verifier lr / epsilon distances / conditionnement.")
                SM.reset_command()
                return net

            loss.backward()
            # Clipping de gradient : garde-fou anti-explosion
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_mse = epoch_loss / max(n_batches, 1)

        # ---------------- VALIDATION ----------------
        net.eval()
        with torch.no_grad():
            SM.set_command_sequence(SM.build_command_sequence(cfg["T_train"]))
            X_hat_val = net(Y_val)
            val_loss = mse(X_hat_val, Xtgt_val).item()

        SM.reset_command()

        # dB pour rester homogene avec tes metriques
        train_db = 10.0 * torch.log10(torch.tensor(train_mse)).item()
        val_db = 10.0 * torch.log10(torch.tensor(val_loss)).item()

        # ---------------- CHECKPOINT ----------------
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(net.state_dict(), cfg["save_path"])
            marker = "  <- best (sauvegarde)"

        if epoch % 5 == 0 or marker:
            print(f"[epoch {epoch:3d}] train {train_db:7.2f} dB | "
                  f"val {val_db:7.2f} dB{marker}")

    print(f"\nMeilleure val : {10*torch.log10(torch.tensor(best_val)).item():.2f} dB")
    print(f"Poids sauvegardes dans : {cfg['save_path']}")
    return net


if __name__ == "__main__":
    train()
