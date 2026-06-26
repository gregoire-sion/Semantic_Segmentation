"""
KalmanNet.py
============
Architecture KalmanNet pour le probleme a 3 drones.

Choix d'integration (Option A, comme le depot officiel) : le nn.Module
porte A LA FOIS le reseau (FC + GRU + FC) ET la boucle de filtrage.
Le forward() deroule la recursion prediction/update sur toute la
trajectoire et renvoie les etats estimes.

Architecture #1 du papier : input FC -> GRU -> output FC.
Features utilisees : F2 (innovation difference) et F4 (forward update
difference), combinaison {F2, F4} recommandee dans le papier.

KalmanNet ne maintient PAS de covariance explicite : l'etat cache de la
GRU encode implicitement l'incertitude.
"""

import torch
import torch.nn as nn

import SystemModel as SM


class KalmanNet(nn.Module):
    def __init__(self, m=SM.M, n=SM.N, gru_mult=10, n_gru_layers=2):
        """
        m, n        : dimensions etat / observation
        gru_mult    : multiplicateur de la taille du hidden state.
                      hidden = gru_mult * (m^2 + n^2)  (cf. papier).
                      ATTENTION : avec m=24, m^2=576 -> hidden enorme.
                      Voir note de dimensionnement plus bas.
        n_gru_layers: nombre de couches GRU empilees.
        """
        super().__init__()
        self.m, self.n = m, n

        # --- Dimensionnement du hidden state ---
        # Le papier propose 10*(m^2+n^2). Ici m=24 -> 10*(576+64) = 6400.
        # C'est lourd mais faisable. Tu peux reduire gru_mult a 1-2 pour
        # commencer (debug NaN plus rapide) puis remonter.
        self.h_dim = gru_mult * (m**2 + n**2)
        self.n_gru_layers = n_gru_layers

        # Dimension des features d'entree : F2 (n) + F4 (m)
        in_dim = n + m

        # --- Couche FC d'entree ---
        self.fc_in = nn.Sequential(
            nn.Linear(in_dim, self.h_dim),
            nn.ReLU(),
        )

        # --- GRU ---
        self.gru = nn.GRU(
            input_size=self.h_dim,
            hidden_size=self.h_dim,
            num_layers=n_gru_layers,
            batch_first=True,
        )

        # --- Couche FC de sortie : produit le gain de Kalman aplati (m*n) ---
        self.fc_out = nn.Sequential(
            nn.Linear(self.h_dim, m * n),
        )

        self.hidden = None   # etat cache GRU, initialise par init_hidden()

    # ---------------------------------------------------------------------
    def init_hidden(self, batch, device):
        """Initialise l'etat cache GRU a zero en debut de trajectoire."""
        self.hidden = torch.zeros(
            self.n_gru_layers, batch, self.h_dim, device=device
        )

    # ---------------------------------------------------------------------
    def compute_KG(self, feature):
        """Passe une feature [batch, n+m] dans le reseau -> KG [batch, m, n]."""
        z = self.fc_in(feature)                       # [batch, h_dim]
        z = z.unsqueeze(1)                            # [batch, 1, h_dim]
        out, self.hidden = self.gru(z, self.hidden)   # out [batch,1,h_dim]
        kg = self.fc_out(out.squeeze(1))              # [batch, m*n]
        return kg.reshape(-1, self.m, self.n)         # [batch, m, n]

    # ---------------------------------------------------------------------
    def forward(self, Y):
        """Deroule KalmanNet sur une trajectoire batchee.

        Y : [batch, n, T] observations.
        Retourne X_hat : [batch, m, T] etats estimes a posteriori.
        """
        batch, _, T = Y.shape
        device = Y.device

        self.init_hidden(batch, device)

        # --- Initialisation des etats ---
        x_post = torch.zeros(batch, self.m, 1, device=device)   # x_0
        # Memoires pour les features (valeurs a t-1)
        y_prev = Y[:, :, 0:1].clone()                # y_{t-1} (init : y_0)
        x_post_prev = x_post.clone()                 # x_{t-1|t-1}

        X_hat = torch.zeros(batch, self.m, T, device=device)

        # IMPORTANT : la commande doit etre rejouee a chaque pas. Elle est
        # supposee deja injectee via SM.set_command_sequence(...) par
        # l'appelant (Train/Simulation), comme dans le data-gen.
        for t in range(T):
            y_t = Y[:, :, t:t+1]                      # [batch, n, 1]

            # ---- Prediction (1er moment seulement) ----
            x_prior = SM.f(x_post, t)                 # x_{t|t-1}
            y_prior = SM.h(x_prior)                   # y_{t|t-1}

            # ---- Features ----
            # F2 : innovation difference = y_t - y_{t|t-1}
            innov = (y_t - y_prior).squeeze(-1)       # [batch, n]
            # F4 : forward update difference = x_{t|t} - x_{t|t-1}
            #      indisponible a t (x_post pas encore calcule) -> on
            #      utilise la valeur a t-1 : x_post_prev - x_prior_prev.
            #      Approche standard du papier : on passe (x_post - x_prior)
            #      du pas PRECEDENT. On le reconstruit ci-dessous.
            fwd_upd = (x_post - x_prior).squeeze(-1)  # [batch, m] (du pas courant prior vs post precedent)

            feature = torch.cat([innov, fwd_upd], dim=1)   # [batch, n+m]

            # ---- Gain de Kalman appris ----
            KG = self.compute_KG(feature)             # [batch, m, n]

            # ---- Update ----
            x_post = x_prior + KG @ (y_t - y_prior)   # [batch, m, 1]

            X_hat[:, :, t] = x_post[:, :, 0]

            # ---- Mise a jour des memoires ----
            y_prev = y_t.clone()
            x_post_prev = x_post.clone()

        return X_hat


if __name__ == "__main__":
    # Smoke test (necessite torch). Verifie juste que le forward tourne
    # et que les dimensions sont bonnes.
    torch.manual_seed(0)
    net = KalmanNet(gru_mult=1)   # gru_mult=1 pour un test leger
    print(f"Hidden dim : {net.h_dim}")
    print(f"Nb parametres : {sum(p.numel() for p in net.parameters()):,}")

    SM.set_command_sequence(SM.build_command_sequence(T=20))
    Y = torch.randn(2, SM.N, 20)
    X_hat = net(Y)
    print(f"X_hat shape : {tuple(X_hat.shape)}  (attendu [2, {SM.M}, 20])")
    print(f"X_hat fini : {torch.isfinite(X_hat).all().item()}")
    SM.reset_command()
