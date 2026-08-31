"""
Scenarios de decalage distributionnel pour l'etude de generalisation.

Module purement declaratif : il decrit QUOI tester, jamais COMMENT.
L'execution est dans eval_generalization.py.

Principe de construction
------------------------
Chaque axe possede exactement un scenario de reference (in_distribution=True)
qui reproduit les conditions d'entrainement. Ce point de reference est le MEME
sur tous les axes (SCENARIO_REF ci-dessous) : c'est ce qui permet de comparer
les axes entre eux, et de verifier que neuf chemins de code differents
convergent bien vers la meme condition nominale.

La famille de commande est figee a FAMILY_REF sur tous les axes autres que
"commandes" et "amplitude" : sans cela on melangerait deux decalages a la fois
et l'effet mesure ne serait attribuable a rien.
"""

from dataclasses import dataclass, field

from KalmanNet_Drones import CAPTEURS_GPS
from ood_commands import FAMILY_REF, FAMILIES_TEST, LABELS_FR

DT = 0.1          # pas de temps du modele (SystemModel.dt)
T_REF = 160       # horizon d'entrainement (CFG.T)


@dataclass(frozen=True)
class Scenario:
    """Une condition de test unique.

    name    : identifiant court, sert de cle JSON et de graine.
    axis    : axe auquel il appartient.
    label   : libelle lisible pour les figures et le tableau.
    x       : abscisse numerique quand l'axe est un balayage (None sinon).
    sm_kwargs  : passes a SystemModel(...)        -> x0, ratio_gps
    gen_kwargs : passes a generate_trajectory(...) -> T, r_scale, q_scale, offset_scale
    cmd        : passes a build_command(...)       -> kind, A
    corrupt    : passes a corrupt_observations(...) -> outages, outlier_rate
    """
    name: str
    axis: str
    label: str
    x: float = None
    sm_kwargs: dict = field(default_factory=dict)
    gen_kwargs: dict = field(default_factory=dict)
    cmd: dict = field(default_factory=lambda: {"kind": FAMILY_REF, "A": None})
    corrupt: dict = field(default_factory=dict)
    in_distribution: bool = False   # point de reference DE CET AXE
    nominal: bool = False           # strictement les conditions d'entrainement


@dataclass(frozen=True)
class Axe:
    """Un axe de decalage : une liste de scenarios et de quoi les tracer."""
    name: str
    groupe: str          # regroupement pour le rapport (6 axes conceptuels)
    titre: str
    xlabel: str          # "" -> axe categoriel, tracer en barres
    scenarios: tuple


def _ref(axis, label="reference (conditions d'entrainement)", x=None, **extra):
    """Le scenario nominal, decline sur chaque axe avec son abscisse propre."""
    base = dict(
        name=f"{axis}__ref", axis=axis, label=label, x=x,
        in_distribution=True, nominal=True,
        gen_kwargs={"T": T_REF, "r_scale": 1.0, "q_scale": 1.0, "offset_scale": 1.0},
        cmd={"kind": FAMILY_REF, "A": None},
    )
    base.update(extra)
    return Scenario(**base)


def _gen(**kw):
    """gen_kwargs complets : on repart toujours du nominal explicite."""
    base = {"T": T_REF, "r_scale": 1.0, "q_scale": 1.0, "offset_scale": 1.0}
    base.update(kw)
    return base


# --------------------------------------------------------------- 1. commandes
AXE_COMMANDES = Axe(
    name="commandes", groupe="commandes",
    titre="Familles de commandes jamais vues a l'entrainement",
    xlabel="",
    scenarios=(_ref("commandes", label=LABELS_FR[FAMILY_REF]),) + tuple(
        Scenario(name=f"commandes__{k}", axis="commandes", label=LABELS_FR[k],
                 gen_kwargs=_gen(), cmd={"kind": k, "A": None})
        for k in FAMILIES_TEST),
)

# Extrapolation en amplitude : le modele A n'a vu que A ~ U(0.9, 1.1).
# Seul axe dont la reference n'est PAS la condition nominale : elle porte sur la
# famille creneaux, jamais vue a l'entrainement. A = 1 y est le point de
# reference de l'axe (in_distribution), mais pas un point nominal.
A_SWEEP = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
AXE_AMPLITUDE = Axe(
    name="amplitude", groupe="commandes",
    titre="Extrapolation en amplitude de commande (famille creneaux)",
    xlabel="Amplitude de commande A",
    scenarios=tuple(
        Scenario(name=f"amplitude__A{A:g}", axis="amplitude",
                 label=f"A = {A:g}", x=A, gen_kwargs=_gen(),
                 cmd={"kind": "creneaux", "A": A},
                 in_distribution=(A == 1.0))
        for A in A_SWEEP),
)

# ----------------------------------------------------------------- 2. horizon
T_SWEEP = (160, 320, 480, 960)
AXE_HORIZON = Axe(
    name="horizon", groupe="horizon",
    titre="Horizon plus long que celui vu a l'entrainement",
    xlabel="Horizon T (pas)",
    scenarios=(_ref("horizon", label=f"T = {T_REF} (entrainement)", x=T_REF),) + tuple(
        Scenario(name=f"horizon__T{T}", axis="horizon",
                 label=f"T = {T} ({T // T_REF}x)", x=T, gen_kwargs=_gen(T=T))
        for T in T_SWEEP if T != T_REF),
)

# ---------------------------------------------------------------- 3. capteurs
GPS_SWEEP = (1, 2, 5, 10, 20)
AXE_CADENCE = Axe(
    name="cadence", groupe="capteurs",
    titre="Cadence du GPS et des mesures de distance",
    xlabel="ratio_gps (1 mesure tous les N pas)",
    scenarios=(_ref("cadence", label="ratio_gps = 5 (entrainement)", x=5),) + tuple(
        Scenario(name=f"cadence__gps{r}", axis="cadence",
                 label=f"ratio_gps = {r}", x=r, gen_kwargs=_gen(),
                 sm_kwargs={"ratio_gps": r})
        for r in GPS_SWEEP if r != 5),
)

# Panne totale GPS + distances : il ne reste que l'accelerometre.
PANNE_SEC = (0.0, 2.0, 5.0, 10.0)


def _panne(duree_s):
    n = int(round(duree_s / DT))
    k0 = (T_REF - n) // 2
    return ((k0, k0 + n - 1, CAPTEURS_GPS),)


AXE_PANNE = Axe(
    name="panne", groupe="capteurs",
    titre="Panne GPS + distances (seul l'accelerometre subsiste)",
    xlabel="Duree de la panne (s)",
    scenarios=(_ref("panne", label="aucune panne", x=0.0),) + tuple(
        Scenario(name=f"panne__{d:g}s", axis="panne",
                 label=f"panne de {d:g} s", x=d, gen_kwargs=_gen(),
                 corrupt={"outages": _panne(d)})
        for d in PANNE_SEC if d > 0.0),
)

OUTLIER_SWEEP = (0.0, 0.01, 0.02, 0.05)
AXE_ABERRATIONS = Axe(
    name="aberrations", groupe="capteurs",
    titre="Mesures aberrantes (bruit x10 sur une fraction des mesures)",
    xlabel="Taux de mesures aberrantes",
    scenarios=(_ref("aberrations", label="aucune aberration", x=0.0),) + tuple(
        Scenario(name=f"aberrations__{p:g}", axis="aberrations",
                 label=f"{100 * p:g} % aberrantes", x=p, gen_kwargs=_gen(),
                 corrupt={"outlier_rate": p})
        for p in OUTLIER_SWEEP if p > 0.0),
)

# ------------------------------------------------------------------- 4. bruit
# Entrainement du modele B : 1/r^2 tire dans [-10, 30] dB.
# -30/-20 dB et 40/50 dB sont donc hors plage des deux cotes.
R_SWEEP_DB = (-30, -20, -10, 0, 10, 20, 30, 40, 50)
AXE_BRUIT_R = Axe(
    name="bruit_r", groupe="bruit",
    titre="Bruit de mesure, y compris hors de la plage d'entrainement",
    xlabel=r"$1/r^2$ [dB]",
    scenarios=tuple(
        Scenario(name=f"bruit_r__{db:+d}dB", axis="bruit_r",
                 label=f"{db:+d} dB", x=db,
                 gen_kwargs=_gen(r_scale=10.0 ** (-db / 20.0)),
                 in_distribution=(db == 0), nominal=(db == 0))
        for db in R_SWEEP_DB),
)

Q_SWEEP = (0.2, 1.0, 5.0, 25.0)
AXE_BRUIT_Q = Axe(
    name="bruit_q", groupe="bruit",
    titre="Bruit de process (jamais varie a l'entrainement)",
    xlabel="Facteur sur l'ecart-type du bruit de process",
    scenarios=tuple(
        Scenario(name=f"bruit_q__{q:g}", axis="bruit_q",
                 label=f"q x {q:g}", x=q, gen_kwargs=_gen(q_scale=q),
                 in_distribution=(q == 1.0), nominal=(q == 1.0))
        for q in Q_SWEEP),
)

# -------------------------------------------------------------- 5. geometrie
GEOMETRIES = ("triangle", "ligne", "serree", "large")
LABELS_GEO = {
    "triangle": "triangle (entrainement)",
    "ligne":    "colineaire (d13 = d12 + d23)",
    "serree":   "formation serree",
    "large":    "formation large",
}
AXE_GEOMETRIE = Axe(
    name="geometrie", groupe="geometrie",
    titre="Geometrie de la formation (regime de non-linearite de h)",
    xlabel="",
    scenarios=(_ref("geometrie", label=LABELS_GEO["triangle"]),) + tuple(
        Scenario(name=f"geometrie__{g}", axis="geometrie", label=LABELS_GEO[g],
                 gen_kwargs=_gen(), sm_kwargs={"x0": g})
        for g in GEOMETRIES if g != "triangle"),
)

# ------------------------------------------------------- 6. condition initiale
OFFSET_SWEEP = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)
AXE_CONDITION_INITIALE = Axe(
    name="condition_initiale", groupe="condition_initiale",
    titre="Erreur sur l'etat initial (les filtres partent toujours de x0 nominal)",
    xlabel="Amplitude de la perturbation initiale (x chol(P0))",
    scenarios=tuple(
        Scenario(name=f"condition_initiale__{s:g}", axis="condition_initiale",
                 label=f"offset x {s:g}", x=s, gen_kwargs=_gen(offset_scale=s),
                 in_distribution=(s == 1.0), nominal=(s == 1.0))
        for s in OFFSET_SWEEP),
)


AXES = (
    AXE_COMMANDES, AXE_AMPLITUDE,
    AXE_HORIZON,
    AXE_CADENCE, AXE_PANNE, AXE_ABERRATIONS,
    AXE_BRUIT_R, AXE_BRUIT_Q,
    AXE_GEOMETRIE,
    AXE_CONDITION_INITIALE,
)

AXES_PAR_NOM = {a.name: a for a in AXES}
GROUPES = ("commandes", "horizon", "capteurs", "bruit", "geometrie",
           "condition_initiale")


if __name__ == "__main__":
    noms = set()
    nominaux = []
    for a in AXES:
        refs = [s for s in a.scenarios if s.in_distribution]
        assert len(refs) == 1, (a.name, len(refs))
        nominaux += [s for s in a.scenarios if s.nominal]
        for s in a.scenarios:
            assert s.name not in noms, f"nom duplique : {s.name}"
            noms.add(s.name)
            assert s.axis == a.name, (s.name, s.axis, a.name)
        print(f"{a.name:20s} groupe={a.groupe:18s} {len(a.scenarios)} scenarios "
              f"| ref = {refs[0].label}"
              + ("" if refs[0].nominal else "  (non nominale)"))

    # Tous les scenarios nominaux doivent decrire EXACTEMENT la meme condition.
    empreintes = {(tuple(sorted(s.sm_kwargs.items())),
                   tuple(sorted(s.gen_kwargs.items())),
                   tuple(sorted(s.cmd.items())),
                   tuple(sorted(s.corrupt.items()))) for s in nominaux}
    assert len(empreintes) == 1, f"conditions nominales divergentes : {empreintes}"
    print(f"\n{len(AXES)} axes, {len(noms)} scenarios, {len(nominaux)} nominaux "
          f"(condition identique), {len(GROUPES)} groupes.")
