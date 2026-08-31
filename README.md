# KalmanNet 3 drones — étude de généralisation

Estimation d'état d'une formation de 3 drones par KalmanNet, comparée à un EKF.
Ce dépôt contient le modèle, son entraînement, et un banc d'essai qui mesure
**jusqu'où l'estimateur appris tient quand les conditions de test s'éloignent
des conditions d'entraînement**.

## Le problème posé

État : 24 composantes (3 drones × `[x, y, vx, vy, ax, ay, bx, by]`).
Mesures : 7 composantes — GPS du drone 1, accéléromètre du drone 2 (biaisé), et
les 3 distances inter-drones. L'observation est non linéaire (d'où l'EKF), et
les mesures sont intermittentes : IMU à chaque pas, GPS et distances tous les
5 pas.

Le filtre est **volontairement mal spécifié sur le drone 2** :

| | vérité | filtre |
|---|---|---|
| accélération drone 2 | pilotée par la commande | supposée constante (`F_filter`) |
| commande drone 2 | connue (`B_true`) | inconnue (`B_filter`, `estim_d2=False`) |
| σ accéléromètre | 0.01 (`R_gen`) | 0.1 (`R`) |

Le drone 2 n'a donc ni GPS, ni commande connue du filtre, ni modèle
d'accélération correct. C'est là que KalmanNet peut apporter quelque chose que
l'EKF ne peut pas — et c'est la position du drone 2 qui sert de métrique.

## Métrique

```
Δ_dB = 10 · log10( MSE_KNet / MSE_EKF )     Δ < 0 → KalmanNet bat l'EKF
div  = fraction des runs où MSE_KNet > 100 × MSE_EKF
```

La comparaison est **appariée** : pour un scénario donné, les mêmes
trajectoires servent à l'EKF et à tous les modèles. On raisonne en ratio et
jamais en MSE absolue — quand un scénario devient plus dur, l'EKF se dégrade
aussi, et une MSE absolue ferait croire à un mérite qui vient en fait de la
difficulté du cas.

## Protocole : deux modèles

Pour distinguer ce qui vient de l'architecture de ce qui vient de la diversité
d'entraînement, deux modèles sont entraînés **avec exactement les mêmes
hyperparamètres et les mêmes graines** :

| modèle | commandes vues | bruit de mesure vu | sortie |
|---|---|---|---|
| **A (étroit)** | `phases3_ref` seule | fixe | `./Dataset/` |
| **B (randomisé)** | `phases3_rand`, `ou` | balayé sur `[-10, 30]` dB | `./Dataset_ood/` |

Chacun est entraîné dans les deux architectures du papier KalmanNet :
`archi1` (un seul GRU, features F2/F4) et `archi2` (trois GRU Q/Σ/S,
features F1–F4, GRU initialisés par les priors `Q`, `P0`, `R`).

## Les six axes de décalage

Chaque axe possède un scénario de référence qui reproduit les conditions
d'entraînement. Les neuf scénarios *nominaux* décrivent une condition
**structurellement identique** (garanti par test, pas par convention) : c'est
ce qui rend les axes comparables entre eux.

| groupe | ce qui varie |
|---|---|
| **commandes** | 4 familles jamais vues (`creneaux`, `chirp`, `virage`, `stopgo`), puis extrapolation en amplitude `A ∈ [0.5, 3]` |
| **horizon** | `T ∈ {160, 320, 480, 960}` — entraîné à 160. Faiblesse classique : la mémoire GRU dérive au-delà de l'horizon vu |
| **capteurs** | cadence GPS `ratio_gps ∈ {1, 2, 5, 10, 20}` ; panne totale GPS + distances de 2/5/10 s ; 1–5 % de mesures aberrantes |
| **bruit** | `1/r² ∈ [-30, +50]` dB, soit ±20 dB **hors** de la plage d'entraînement des deux côtés ; bruit de process `× {0.2, 1, 5, 25}`, jamais varié à l'entraînement |
| **géométrie** | formation `triangle` (réf), `colinéaire` (jacobienne de `h` déficiente en rang), `serrée`, `large` |
| **condition initiale** | perturbation de l'état initial `× {0, 0.5, 1, 2, 3, 5}` ; les filtres partent toujours du `x0` nominal |

Le Δ_dB est calculé sur cinq fenêtres temporelles, pour voir ce qu'une moyenne
globale masquerait :

| fenêtre | ce qu'elle mesure |
|---|---|
| `total` | toute la trajectoire — c'est le Δ_dB rapporté dans le tableau |
| `transitoire` | les 2 premières secondes : reconvergence après une erreur d'état initial |
| `etabli` | les 2 dernières secondes : régime permanent |
| `bloc_debut` | les 160 premiers pas |
| `bloc_fin` | les 160 derniers pas |

Le rapport entre `bloc_fin` et `bloc_debut` mesure la **dérive** quand `T` dépasse
l'horizon vu à l'entraînement : si le réseau décroche au-delà de 160 pas, l'écart
entre ces deux fenêtres le montre directement.

## Utilisation

```bash
pip install -r requirements.txt

python tests_generalisation.py       # vérifie l'outillage (~1 min)
python train_models.py               # entraîne A et B × 2 architectures
python etude_generalisation.py       # cartographie les 51 scénarios
```

Ajouter `--fumee` à l'un des deux derniers pour une version jouet qui valide le
code sans valeur scientifique. L'évaluation complète compte 51 scénarios ×
`N_MC = 20` trajectoires ; les scénarios `T = 960` dominent le temps de calcul.

Sorties dans `./eval_generalisation/` : `tableau.md` (Δ_dB prêt à coller dans un
rapport), `resultats.json`, une figure par axe, une figure de synthèse, et des
séries temporelles sur les cas les plus parlants.

## Fichiers

| fichier | rôle |
|---|---|
| `KalmanNet_Drones.py` | modèle système, EKF, KalmanNet, entraînement, tracés |
| `ood_commands.py` | 7 familles de commandes (`build_command`) |
| `etude_generalisation.py` | l'étude complète : scénarios, évaluation, figures |
| `train_models.py` | entraîne les modèles A et B |
| `eval_ood.py` | banc d'essai historique, axe commandes seul (référence de non-régression) |
| `tests_generalisation.py` | tests, dont la non-régression bit-à-bit de `generate_trajectory` |
| `KalmanNet_Drones_origine.py` | copie figée d'avant modification, servant de référence aux tests — ne pas modifier |
| `generate_dataset.py` | génération hors ligne d'un dataset avec split disjoint |
| `test.py` | évaluation sur une trajectoire à état initial décalé |

## Notes de conception

- Les paramètres ajoutés à `generate_trajectory` (`T`, `q_scale`,
  `offset_scale`) et à `SystemModel` (`x0`, `ratio_gps`, `ratio_imu`) ont des
  défauts qui reproduisent **bit à bit** le comportement d'origine — vérifié par
  `tests_generalisation.py` sur 12 combinaisons de réglages. Les checkpoints
  entraînés avant cette modification restent valides.
- `KalmanNetNN` capture `sm` à la construction (`self.f`, `self.h`,
  `self.prior_*`). Un scénario qui change le modèle système doit passer par
  `rebind(model, sm)`.
- `KalmanNetNN.__init__` et `monte_carlo_knet` lisent `CFG.IN_MULT`,
  `CFG.OUT_MULT` et `CFG.N_MC` comme **valeurs par défaut d'arguments, évaluées
  à l'import** : muter `CFG` après coup ne les change pas. Tout balayage
  d'architecture doit passer ces valeurs explicitement.
- Chaque scénario tire ses trajectoires avec la graine `SEED_EVAL + numero`, où
  `numero` est son rang dans la liste. Surtout pas `hash(nom)` : Python randomise
  le hachage des chaînes à chaque lancement, deux exécutions ne donneraient pas
  les mêmes chiffres.
- `etude_generalisation.py` est volontairement écrit sans `dataclass`, sans
  générateur et sans compréhension imbriquée : il se lit de haut en bas, dans le
  même style que `KalmanNet_Drones.py`. Un scénario est un simple dictionnaire,
  et les valeurs par défaut de la fonction `scenario()` **sont** les conditions
  d'entraînement — un scénario ne mentionne que ce qu'il fait varier.
- Le README a servi de bloc-notes de code pendant 38 révisions avant d'être
  vidé. `ood_commands.py` et `eval_ood.py` ont été restaurés depuis le commit
  `f438312`. Les sources ne reviennent pas ici.

## Ce que cette étude ne fait pas

Elle **cartographie** la dégradation, elle ne cherche pas encore à la corriger :
pas de modèle augmenté sur les nouveaux axes, pas de régularisation. KalmanNet
ne fournit par ailleurs aucune covariance — `nci_coverage` compare un σ
empirique Monte-Carlo au `P` de l'EKF, mais l'absence d'incertitude produite par
le réseau reste une limite de l'approche.
