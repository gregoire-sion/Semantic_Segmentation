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
