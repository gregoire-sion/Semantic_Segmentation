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
