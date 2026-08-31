"""
Cartographie de la generalisation de KalmanNet.

Reprend la metrique etablie par eval_ood.py et l'applique aux six axes de
decalage decrits dans scenarios.py :

    Delta_dB = 10 log10( MSE_KNet / MSE_EKF )     < 0 -> KalmanNet bat l'EKF
    div_rate = fraction des runs ou MSE_KNet > 100 x MSE_EKF

La comparaison est APPARIEE : pour un scenario donne, les memes trajectoires
servent a l'EKF et a tous les modeles. On raisonne en ratio et jamais en MSE
absolue -- quand un scenario devient plus dur, l'EKF se degrade aussi, et une
MSE absolue ferait croire a un merite qui vient en fait de la difficulte du cas.

Usage :
    python eval_generalization.py            # evaluation complete
    python eval_generalization.py --fumee    # version rapide, pour valider le code
"""

import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from KalmanNet_Drones import (CFG, SystemModel, EKF, KalmanNetNN,
                              generate_trajectory, corrupt_observations,
                              run_knet, rebind)
from ood_commands import build_command
from scenarios import AXES, AXES_PAR_NOM, GROUPES, T_REF


# ----------------------------------------------------------------- reglages
MODELS = {                          # nom affiche -> dossier de checkpoints
    "A (etroit)":    "./Dataset",
    "B (randomise)": "./Dataset_ood",
}
ARCHIS    = ["archi1", "archi2"]
N_MC      = 20                      # trajectoires par scenario
SEED_EVAL = 12345
OUT_DIR   = "./eval_generalisation"

DIVERGENCE_FACTOR = 100.0           # run divergent si MSE_KNet > 100 x MSE_EKF
TRANSITOIRE_S     = 2.0             # duree des fenetres transitoire / etabli

# Drone 2 = le drone critique : ni GPS, ni commande connue du filtre
# (B_filter, estim_d2=False), et hypothese d'acceleration constante (F_filter).
IDX_POS_D2  = slice(8, 10)
IDX_POS_ALL = [0, 1, 8, 9, 16, 17]

COULEURS = {("A (etroit)", "archi1"):    "#c44e52",
            ("A (etroit)", "archi2"):    "#8c2d30",
            ("B (randomise)", "archi1"): "#4c72b0",
            ("B (randomise)", "archi2"): "#2a4a7f"}
STYLES = {"archi1": "--", "archi2": "-"}


# -------------------------------------------------------------------- outils
class CacheSysteme:
    """Un SystemModel + EKF par configuration, construits une seule fois.

    Les axes geometrie et cadence changent le SystemModel ; le reconstruire a
    chaque trajectoire couterait cher pour rien.
    """

    def __init__(self):
        self._cache = {}

    def get(self, sm_kwargs):
        cle = tuple(sorted(sm_kwargs.items()))
        if cle not in self._cache:
            sm = SystemModel(**sm_kwargs)
            self._cache[cle] = (sm, EKF(sm))
        return self._cache[cle]


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
    return {n: d for n, d in out.items() if d}


def _mse(xa, xb, idx, deb=None, fin=None):
    """MSE sur un sous-ensemble de composantes, et optionnellement de temps."""
    d = (xa[deb:fin, idx, 0] - xb[deb:fin, idx, 0])
    return (d ** 2).mean().item()


def _fenetres(T, dt):
    """Decoupage temporel utilise pour les metriques fines.

    - blocs de T_REF pas : revele la derive sur l'axe horizon, invisible dans
      un Delta_dB global qui moyenne tout ;
    - transitoire / etabli : quantifie la reconvergence apres une panne GPS ou
      une erreur d'etat initial.
    """
    n_trans = max(1, int(round(TRANSITOIRE_S / dt)))
    fen = {"transitoire": (0, n_trans + 1), "etabli": (T + 1 - n_trans, T + 1)}
    if T > T_REF:
        for i in range(T // T_REF):
            fen[f"bloc{i}"] = (i * T_REF, min((i + 1) * T_REF, T) + 1)
    return fen


def _db(num, den):
    if den <= 0 or num <= 0 or not np.isfinite(num) or not np.isfinite(den):
        return None
    return float(10.0 * np.log10(num / den))


# ------------------------------------------------------------ coeur du calcul
def eval_scenario(cache, models, sc, n_mc, seed):
    """Evalue tous les modeles sur n_mc trajectoires d'un scenario."""
    sm, ekf = cache.get(sc.sm_kwargs)
    for per_archi in models.values():
        for model in per_archi.values():
            rebind(model, sm)

    T = int(sc.gen_kwargs.get("T", CFG.T))
    fen = _fenetres(T, sm.dt)
    rng = np.random.default_rng(seed)

    ekf_acc = {"d2": [], "all": []}
    ekf_acc.update({f"d2_{f}": [] for f in fen})
    acc = {n: {a: {"d2": [], "all": [], "div": 0,
                   **{f"d2_{f}": [] for f in fen}}
               for a in models[n]} for n in models}

    for _ in range(n_mc):
        u = build_command(T, sm.dt, rng, **sc.cmd)
        X, Y, U, M = generate_trajectory(sm, rng, u_seq=u, **sc.gen_kwargs)
        if sc.corrupt:
            Y, M = corrupt_observations(Y, M, rng, sm, **sc.corrupt)

        xe, _ = ekf.run(Y, U, M)
        e_d2 = _mse(xe, X, IDX_POS_D2)
        ekf_acc["d2"].append(e_d2)
        ekf_acc["all"].append(_mse(xe, X, IDX_POS_ALL))
        for f, (a, b) in fen.items():
            ekf_acc[f"d2_{f}"].append(_mse(xe, X, IDX_POS_D2, a, b))

        for name, per_archi in models.items():
            for archi, model in per_archi.items():
                xk = run_knet(sm, model, Y, U, M)
                k_d2 = _mse(xk, X, IDX_POS_D2)
                a_ = acc[name][archi]
                if (not np.isfinite(k_d2)) or k_d2 > DIVERGENCE_FACTOR * e_d2:
                    a_["div"] += 1
                if not np.isfinite(k_d2):
                    continue
                a_["d2"].append(k_d2)
                a_["all"].append(_mse(xk, X, IDX_POS_ALL))
                for f, (a, b) in fen.items():
                    a_[f"d2_{f}"].append(_mse(xk, X, IDX_POS_D2, a, b))

    res = {"label": sc.label, "x": sc.x, "T": T,
           "in_distribution": sc.in_distribution, "nominal": sc.nominal,
           "ekf_d2": float(np.mean(ekf_acc["d2"])),
           "ekf_all": float(np.mean(ekf_acc["all"])),
           "modeles": {}}
    for name in models:
        res["modeles"][name] = {}
        for archi in models[name]:
            a_ = acc[name][archi]
            if not a_["d2"]:
                res["modeles"][name][archi] = None
                continue
            m_d2 = float(np.mean(a_["d2"]))
            entree = {"mse_d2": m_d2,
                      "mse_all": float(np.mean(a_["all"])),
                      "delta_d2": _db(m_d2, res["ekf_d2"]),
                      "delta_all": _db(float(np.mean(a_["all"])), res["ekf_all"]),
                      "div_rate": a_["div"] / n_mc}
            for f in fen:
                entree[f"delta_{f}"] = _db(float(np.mean(a_[f"d2_{f}"])),
                                           float(np.mean(ekf_acc[f"d2_{f}"])))
            res["modeles"][name][archi] = entree
    return res


def eval_axe(cache, models, axe, n_mc, seed_base):
    out = {"titre": axe.titre, "xlabel": axe.xlabel, "groupe": axe.groupe,
           "scenarios": {}}
    for sc in axe.scenarios:
        # Graine derivee du nom : chaque scenario tire ses propres trajectoires,
        # donc les references des differents axes sont des estimations
        # independantes de la meme quantite (cf. controle de coherence).
        graine = seed_base + (abs(hash(sc.name)) % 100000)
        out["scenarios"][sc.name] = eval_scenario(cache, models, sc, n_mc, graine)
        r = out["scenarios"][sc.name]
        bouts = []
        for name in models:
            for archi in sorted(models[name]):
                e = r["modeles"][name][archi]
                bouts.append(f"{name[0]}-{archi[-1]}:"
                             + ("  n/a" if e is None
                                else f"{e['delta_d2']:+6.2f}dB"
                                     + (f"({100 * e['div_rate']:.0f}%div)"
                                        if e["div_rate"] else "")))
        print(f"   {sc.label:44s} EKF={r['ekf_d2']:9.3f} | " + "  ".join(bouts))
    return out


# ------------------------------------------------------------------- figures
def _series(resultats_axe, models):
    """[(nom_modele, archi, [Delta_dB par scenario])] dans l'ordre des scenarios."""
    noms = list(resultats_axe["scenarios"])
    for name in models:
        for archi in sorted(models[name]):
            vals = []
            for n in noms:
                e = resultats_axe["scenarios"][n]["modeles"][name][archi]
                vals.append(np.nan if e is None or e["delta_d2"] is None
                            else e["delta_d2"])
            yield name, archi, np.array(vals)


def plot_axe(resultats_axe, axe, models, outdir):
    scs = resultats_axe["scenarios"]
    noms = list(scs)
    labels = [scs[n]["label"] for n in noms]
    fig, ax = plt.subplots(figsize=(max(8.5, 1.6 * len(noms) + 3), 5.5))

    if axe.xlabel:                                   # axe numerique -> courbes
        xs = [scs[n]["x"] for n in noms]
        for name, archi, vals in _series(resultats_axe, models):
            ax.plot(xs, vals, "o" + STYLES[archi], lw=2, ms=5,
                    color=COULEURS.get((name, archi)),
                    label=f"{name} / {archi}")
        ax.set_xlabel(axe.xlabel)
        for n in noms:
            if scs[n]["in_distribution"]:
                ax.axvline(scs[n]["x"], color="grey", ls=":", lw=1.5)
                ax.text(scs[n]["x"], ax.get_ylim()[1], " entrainement",
                        fontsize=8, ha="left", va="top", color="dimgrey")
    else:                                            # axe categoriel -> barres
        series = list(_series(resultats_axe, models))
        xs = np.arange(len(noms))
        width = 0.8 / max(len(series), 1)
        for i, (name, archi, vals) in enumerate(series):
            ax.bar(xs + i * width - 0.4 + width / 2, vals, width,
                   color=COULEURS.get((name, archi)),
                   label=f"{name} / {archi}")
        ax.set_xticks(xs)
        ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=9)

    ax.axhline(0, color="k", lw=1.2)
    ax.set_ylabel(r"$\Delta_{dB} = 10\log_{10}(MSE_{KNet}/MSE_{EKF})$")
    ax.set_title(f"{axe.titre}\nen dessous de 0 : KalmanNet bat l'EKF",
                 fontsize=12)
    ax.grid(True, axis="y", ls=":", alpha=.7)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(outdir, f"axe_{axe.name}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_synthese(resultats, models, outdir):
    """Pire Delta_dB par groupe : la vue d'ensemble de la degradation."""
    series = {}
    for name in models:
        for archi in sorted(models[name]):
            pires = []
            for g in GROUPES:
                vals = [e["delta_d2"]
                        for a in resultats.values() if a["groupe"] == g
                        for s in a["scenarios"].values()
                        for e in [s["modeles"][name][archi]]
                        if e and e["delta_d2"] is not None]
                pires.append(max(vals) if vals else np.nan)
            series[(name, archi)] = pires

    xs = np.arange(len(GROUPES))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, ((name, archi), vals) in enumerate(series.items()):
        ax.bar(xs + i * width - 0.4 + width / 2, vals, width,
               color=COULEURS.get((name, archi)), label=f"{name} / {archi}")
    ax.axhline(0, color="k", lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([g.replace("_", "\n") for g in GROUPES], fontsize=9)
    ax.set_ylabel(r"pire $\Delta_{dB}$ de l'axe")
    ax.set_title("Synthese : degradation dans le cas le plus defavorable "
                 "de chaque axe", fontsize=12)
    ax.grid(True, axis="y", ls=":", alpha=.7)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(outdir, "synthese.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_serie_temporelle(cache, models, sc, outdir, seed=999):
    """Erreur de position du drone 2 au cours du temps, sur une trajectoire."""
    sm, ekf = cache.get(sc.sm_kwargs)
    for per_archi in models.values():
        for model in per_archi.values():
            rebind(model, sm)

    T = int(sc.gen_kwargs.get("T", CFG.T))
    rng = np.random.default_rng(seed)
    u = build_command(T, sm.dt, rng, **sc.cmd)
    X, Y, U, M = generate_trajectory(sm, rng, u_seq=u, **sc.gen_kwargs)
    if sc.corrupt:
        Y, M = corrupt_observations(Y, M, rng, sm, **sc.corrupt)
    temps = np.arange(T + 1) * sm.dt

    def err(x):
        # float64 : sur un modele divergent la norme deborde en float32.
        d = (x[:, 8:10, 0] - X[:, 8:10, 0]).cpu().numpy().astype(np.float64)
        return np.linalg.norm(d, axis=1)

    xe, _ = ekf.run(Y, U, M)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(temps, err(xe), color="green", lw=1.6, label="EKF")
    for name, per_archi in models.items():
        for archi, model in per_archi.items():
            ax.plot(temps, err(run_knet(sm, model, Y, U, M)), lw=1.5,
                    ls=STYLES[archi], color=COULEURS.get((name, archi)),
                    label=f"{name} / {archi}")
    for k0, k1, _cap in sc.corrupt.get("outages", ()):
        ax.axvspan(k0 * sm.dt, k1 * sm.dt, color="red", alpha=.12)
        ax.text((k0 + k1) / 2 * sm.dt, ax.get_ylim()[1], "panne GPS",
                fontsize=8, ha="center", va="top", color="firebrick")

    ax.set_yscale("log")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Erreur de position drone 2 (m)")
    ax.set_title(f"Reponse temporelle — {sc.label}", fontsize=12)
    ax.grid(True, ls=":", alpha=.7)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(outdir, f"temporel_{sc.name}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ------------------------------------------------------------------- tableau
def ecrire_tableau(resultats, models, outdir, n_mc):
    colonnes = [(n, a) for n in models for a in sorted(models[n])]
    lignes = ["# Cartographie de la generalisation — Delta_dB",
              "",
              "`Delta_dB = 10 log10(MSE_KNet / MSE_EKF)` sur la position du "
              "drone 2. **Negatif = KalmanNet bat l'EKF.**",
              f"Comparaison appariee, {n_mc} trajectoires par scenario. "
              "`div` = part des runs ou MSE_KNet depasse 100x celle de l'EKF.",
              ""]
    for g in GROUPES:
        lignes += [f"## {g.replace('_', ' ')}", ""]
        for nom_axe, axe_res in resultats.items():
            if axe_res["groupe"] != g:
                continue
            lignes += [f"### {axe_res['titre']}", "",
                       "| scenario | MSE EKF | "
                       + " | ".join(f"{n} / {a}" for n, a in colonnes) + " |",
                       "|---|---|" + "---|" * len(colonnes)]
            for s in axe_res["scenarios"].values():
                cellules = []
                for n, a in colonnes:
                    e = s["modeles"][n][a]
                    if e is None or e["delta_d2"] is None:
                        cellules.append("n/a")
                    else:
                        c = f"{e['delta_d2']:+.2f}"
                        if e["div_rate"]:
                            c += f" ({100 * e['div_rate']:.0f}% div)"
                        cellules.append(c)
                marque = " *(ref)*" if s["in_distribution"] else ""
                lignes.append(f"| {s['label']}{marque} | {s['ekf_d2']:.3f} | "
                              + " | ".join(cellules) + " |")
            lignes.append("")
    p = os.path.join(outdir, "tableau.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lignes))
    return p


def controle_coherence(resultats, models):
    """Les scenarios nominaux decrivent la MEME condition d'entrainement.

    scenarios.py garantit deja qu'ils sont structurellement identiques ; ici on
    verifie la consequence statistique : estimes sur des trajectoires
    independantes, leurs Delta_dB doivent coincider a la variance Monte-Carlo
    pres. Une etendue importante signale un N_MC trop faible pour conclure, ou
    un modele si instable que sa reference elle-meme n'est pas reproductible.

    L'axe amplitude est exclu : sa reference (creneaux a A = 1) est le point
    in-distribution de cet axe, mais pas une condition nominale.
    """
    print("\n== Controle de coherence des scenarios nominaux ==")
    ok = True
    for name in models:
        for archi in sorted(models[name]):
            vals = [s["modeles"][name][archi]["delta_d2"]
                    for a in resultats.values()
                    for s in a["scenarios"].values()
                    if s["nominal"] and s["modeles"][name][archi]
                    and s["modeles"][name][archi]["delta_d2"] is not None]
            if len(vals) < 2:
                continue
            etendue = max(vals) - min(vals)
            statut = "OK " if etendue < 3.0 else "!! "
            ok &= etendue < 3.0
            print(f"   {statut}{name} / {archi} : {len(vals)} nominaux, "
                  f"Delta_dB de {min(vals):+.2f} a {max(vals):+.2f} "
                  f"(etendue {etendue:.2f} dB)")
    if not ok:
        print("   !! etendue > 3 dB : conditions nominales identiques mais "
              "resultats disperses -> augmenter N_MC, ou modele instable.")
    return ok


# ---------------------------------------------------------------------- main
def main(n_mc=N_MC, outdir=OUT_DIR, noms_axes=None):
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(CFG.SEED)
    np.random.seed(CFG.SEED)

    # L'evaluation impose toujours ses propres commandes via u_seq ; on
    # neutralise le flag pour qu'aucun appel indirect ne rebascule dessus.
    CFG.TRAIN_CMD_RANDOMIZE = False

    cache = CacheSysteme()
    sm_ref, _ = cache.get({})
    print(f">> Device : {sm_ref.device}")
    print("== Chargement des modeles ==")
    models = load_models(sm_ref)
    if not models:
        raise SystemExit(
            "Aucun checkpoint trouve. Lancez d'abord : python train_models.py")

    axes = [a for a in AXES if noms_axes is None or a.name in noms_axes]
    resultats = {}
    for axe in axes:
        print(f"\n== Axe : {axe.titre} ==")
        resultats[axe.name] = eval_axe(cache, models, axe, n_mc, SEED_EVAL)

    produced = []
    for axe in axes:
        produced.append(plot_axe(resultats[axe.name], axe, models, outdir))

    # Series temporelles la ou la dynamique compte plus que la moyenne.
    for nom_axe, choix in (("panne", -1), ("horizon", -1),
                           ("condition_initiale", -1)):
        if nom_axe in AXES_PAR_NOM and (noms_axes is None or nom_axe in noms_axes):
            produced.append(plot_serie_temporelle(
                cache, models, AXES_PAR_NOM[nom_axe].scenarios[choix], outdir))

    produced.append(plot_synthese(resultats, models, outdir))
    produced.append(ecrire_tableau(resultats, models, outdir, n_mc))

    chemin_json = os.path.join(outdir, "resultats.json")
    with open(chemin_json, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"n_mc": n_mc, "seed": SEED_EVAL,
                            "modeles": {n: sorted(d) for n, d in models.items()},
                            "divergence_factor": DIVERGENCE_FACTOR},
                   "axes": resultats}, fh, indent=2)
    produced.append(chemin_json)

    controle_coherence(resultats, models)

    print("\n== Sorties ==")
    for p in produced:
        print("  ", p)
    return resultats


if __name__ == "__main__":
    if "--fumee" in sys.argv:
        main(n_mc=2, outdir=os.path.join(OUT_DIR, "fumee"))
    else:
        main()
