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

## Résultats

Étude complète : 51 scénarios, `N_MC = 20`, quatre estimateurs appris
(A/B × archi1/archi2) comparés à l'EKF sur les mêmes trajectoires.

### Le plancher de bruit, à lire avant tout le reste

Les 9 scénarios *nominaux* décrivent la même condition d'entraînement. Ils en
donnent donc 9 estimations indépendantes, et leur dispersion mesure directement
la précision de l'étude :

| modèle | Δ_dB min | Δ_dB max | étendue | écart-type |
|---|---|---|---|---|
| A / archi1 | +1.18 | +13.76 | 12.6 dB | 4.9 dB |
| A / archi2 | −2.06 | +2.90 | 5.0 dB | 1.7 dB |
| B / archi1 | +1.92 | +15.47 | 13.6 dB | 4.3 dB |
| B / archi2 | −2.31 | +4.60 | 6.9 dB | 2.2 dB |

La MSE de l'EKF elle-même varie d'un facteur 2,4 sur ces conditions identiques.

**Conséquence : avec `N_MC = 20`, un écart inférieur à ~5 dB (archi2) ou ~10 dB
(archi1) n'est pas interprétable.** Les effets rapportés ci-dessous dépassent
tous ce seuil. Pour conclure sur des différences plus fines, il faut augmenter
`N_MC`.

Second avertissement : quand `div_rate` est élevé, la moyenne des MSE est
dominée par quelques trajectoires qui explosent, et le Δ_dB perd son sens. Au
delà de ~30 % de divergence, c'est le taux lui-même qu'il faut lire, pas le Δ_dB.

### Synthèse — pire Δ_dB de chaque groupe

| groupe | A/archi1 | A/archi2 | B/archi1 | B/archi2 |
|---|---|---|---|---|
| commandes | +27.8 | +61.9 | +6.7 | **+4.6** |
| horizon | +308.7 | +114.0 | +162.1 | +289.0 |
| capteurs | +148.4 | +17.2 | +92.0 | +42.6 |
| bruit | +40.8 | +96.3 | +23.2 | +65.3 |
| géométrie | +39.9 | +28.5 | +22.2 | +55.5 |
| condition initiale | +21.9 | +3.3 | +13.4 | +35.9 |

Divergences cumulées : A/archi1 29 scénarios sur 51, A/archi2 17, B/archi1 19,
**B/archi2 8**.

### 1. L'horizon est le point de rupture, et c'est une dérive pure

C'est le résultat principal. Les fenêtres `bloc_debut` et `bloc_fin` le
tranchent sans ambiguïté :

| | A/archi1 | | A/archi2 | | B/archi1 | | B/archi2 | |
|---|---|---|---|---|---|---|---|---|
| | début | fin | début | fin | début | fin | début | fin |
| T = 160 | +3.2 | +3.2 | −1.3 | −1.3 | +2.5 | +2.5 | −1.1 | −1.1 |
| T = 320 | +3.6 | +37.7 | +0.7 | +4.2 | +1.9 | +13.1 | +1.1 | +4.4 |
| T = 480 | +3.4 | +159.2 | −7.0 | +1.3 | +3.6 | +54.8 | −3.2 | +95.0 |
| T = 960 | **−0.0** | **+312.0** | **+0.2** | **+117.3** | **+2.6** | **+165.4** | **−1.1** | **+292.3** |

À T = 960, sur les 160 **premiers** pas les quatre modèles sont exactement au
niveau de leur performance nominale. Sur les 160 **derniers**, ils sont à
+117 dB au mieux. Le transitoire reste à ~0 dB partout, donc ce n'est pas un
problème d'initialisation.

**Le réseau reste aussi bon qu'à l'entraînement dans l'horizon qu'il a vu, et
diverge au-delà.** Aucun des quatre n'y échappe, y compris le meilleur.

### 2. La diversité d'entraînement ne se transfère pas d'un axe à l'autre

Écart médian B − A (négatif = le modèle randomisé est meilleur) :

| axe | archi1 | archi2 |
|---|---|---|
| commandes | **−14.0** | **−3.3** |
| amplitude | **−18.1** | **−17.6** |
| cadence | +0.7 | +0.0 |
| panne | −0.0 | +1.7 |
| aberrations | +1.3 | −0.1 |
| bruit (mesure) | −1.9 | −3.1 |
| bruit (process) | −0.7 | −0.9 |
| géométrie | +1.0 | +1.2 |
| condition initiale | −0.0 | +0.2 |

Le modèle B écrase le modèle A sur l'axe des commandes — jusqu'à −61 dB sur la
famille `virage`, où A/archi2 atteint +61.9 dB. C'est la confirmation nette de
l'intérêt de la randomisation.

Mais ce bénéfice **ne dépasse pas l'axe sur lequel la diversité a été
introduite**. Sur les cinq autres groupes, l'écart médian est compris entre
−1.9 et +1.7 dB, c'est-à-dire dans le bruit. Sur la géométrie colinéaire,
B/archi2 (+55.5) est même nettement pire que A/archi2 (+28.5).

### 3. Plus de mesures casse le réseau, moins de mesures ne le gêne pas

Contre-intuitif, et net sur les quatre modèles :

| ratio_gps | EKF | A/archi1 | A/archi2 | B/archi1 | B/archi2 | divergences |
|---|---|---|---|---|---|---|
| **1** (5× plus qu'à l'entraînement) | 6.59 | **+148.4** | **+17.2** | **+92.0** | **+42.6** | 15 % partout |
| 2 | 9.86 | +45.2 | −2.2 | +13.3 | −0.7 | 0–5 % |
| 5 *(entraînement)* | 15.32 | +1.3 | −0.7 | +2.0 | −0.6 | 0 % |
| 10 | 12.70 | +2.0 | −1.4 | +8.3 | −2.5 | 0 % |
| 20 | 16.19 | +8.5 | +3.3 | +10.7 | +2.1 | 0 % |

Donner au réseau **plus** d'information qu'il n'en a vu le détruit, alors qu'il
encaisse sans peine d'en recevoir quatre fois moins. Les statistiques
d'innovation dépendent de la cadence de mesure, et le gain appris y est calé.

### 4. Le bruit de mesure : effondrement d'un côté seulement

Entraînement sur `1/r² ∈ [−10, 30] dB`.

| 1/r² | EKF | A/archi2 | B/archi2 | divergences (B/archi2) |
|---|---|---|---|---|
| −30 dB *(hors plage)* | 536.6 | +96.3 | +65.3 | **100 %** |
| −20 dB *(hors plage)* | 264.1 | +51.2 | +48.1 | 5 % |
| −10 dB *(bord)* | 54.6 | −3.1 | **−6.1** | 0 % |
| 0 dB *(nominal)* | 12.7 | +0.8 | +1.6 | 0 % |
| +40 dB *(hors plage)* | 3.9 | +10.6 | +4.2 | 0 % |
| +50 dB *(hors plage)* | 5.0 | +8.4 | +3.9 | 0 % |

Des mesures **plus bruitées** que vues à l'entraînement provoquent un
effondrement total (100 % de divergence à −30 dB). Des mesures **plus propres**
ne coûtent que 4 à 10 dB, sans divergence. L'extrapolation n'est pas symétrique.

### 5. Là où KalmanNet bat vraiment l'EKF

Seuls les écarts au-delà du plancher de bruit sont retenus (Δ_dB < −4 dB) :

| scénario | modèle | Δ_dB | MSE EKF | divergences |
|---|---|---|---|---|
| panne GPS de 10 s | A/archi2 | **−8.4** | 210.5 | 0 % |
| panne GPS de 10 s | B/archi2 | −5.7 | 210.5 | 0 % |
| 5 % de mesures aberrantes | A/archi2 | **−6.9** | 49.6 | 0 % |
| 5 % de mesures aberrantes | B/archi2 | −5.1 | 49.6 | 0 % |
| bruit de mesure −10 dB | B/archi2 | −6.1 | 54.6 | 0 % |
| état initial exact (offset 0) | A/archi1 | −11.9 | 4.2 | 0 % |
| état initial exact (offset 0) | B/archi2 | −10.4 | 4.2 | 0 % |

Le point commun : ce sont les situations où les hypothèses figées de l'EKF
coûtent le plus cher — une panne prolongée, des mesures aberrantes que son `R`
constant ne sait pas rejeter. **KalmanNet gagne exactement là où le modèle de
l'EKF est faux**, ce qui est sa promesse. Ces gains sont obtenus sans aucune
divergence.

### 6. La géométrie colinéaire met tout le monde en échec

| formation | EKF | A/archi1 | A/archi2 | B/archi1 | B/archi2 |
|---|---|---|---|---|---|
| triangle *(entraînement)* | 10.5 | +12.2 | +2.4 | +15.5 | +2.1 |
| **colinéaire** | 73.0 | +39.9 *(90 % div)* | +28.5 *(35 %)* | +22.2 *(65 %)* | +55.5 *(65 %)* |
| serrée | 31.2 | +28.9 *(40 %)* | +1.8 | +13.4 | +3.1 |
| large | 6.4 | −0.4 | +0.6 | +0.6 | +0.6 |

Quand les trois drones sont alignés, la jacobienne de `h` perd son rang : les
trois distances ne portent plus que deux informations indépendantes. L'EKF s'y
dégrade d'un facteur 7 mais survit ; les réseaux divergent dans 35 à 90 % des
cas.

### Ce que ces résultats disent

`archi2` domine `archi1` partout, et `B/archi2` est le seul des quatre à rester
utilisable : 8 scénarios sur 51 avec au moins une divergence, contre 29 pour
A/archi1. Dans son domaine d'entraînement — horizon de 160 pas, cadence de
mesure connue, bruit dans la plage vue, formation triangulaire — il fait jeu
égal avec l'EKF et le bat nettement sur les pannes et les aberrations.

En dehors, il casse. Et les cinq modes de rupture identifiés (horizon, cadence
plus rapide, bruit plus fort, géométrie dégénérée, offset initial important)
ne sont pas corrigés par la randomisation des commandes.

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
