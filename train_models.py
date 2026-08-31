"""
Entrainement des deux modeles de l'etude de generalisation.

    A (etroit)    : une seule famille de commande, bruit de mesure fixe.
    B (randomise) : plusieurs familles, bruit de mesure balaye sur [-10, 30] dB.

Rien d'autre ne differe : meme architecture, memes hyperparametres, memes
graines de generation. C'est ce qui rend l'ecart de generalisation observe
attribuable a la diversite d'entrainement, et non a l'architecture.

Usage :
    python train_models.py            # les 2 modeles x les 2 architectures
    python train_models.py --fumee    # config jouet, pour valider le code
"""

import os
import sys

import numpy as np
import torch

from KalmanNet_Drones import (CFG, SystemModel, KalmanNetNN, generate_dataset,
                              save_dataset, train, plot_loss)
from ood_commands import FAMILIES_TRAIN, FAMILY_REF

ARCHIS = ["archi1", "archi2"]

CONFIGS = {
    "A (etroit)": dict(
        dossier="./Dataset",
        familles=(FAMILY_REF,),
        noise_sweep=False,
        commentaire="une famille de commande, bruit fixe"),
    "B (randomise)": dict(
        dossier="./Dataset_ood",
        familles=FAMILIES_TRAIN,
        noise_sweep=True,
        commentaire=f"familles {FAMILIES_TRAIN}, bruit balaye {CFG.TRAIN_NOISE_DB} dB"),
}


def entraine(nom, conf, archis=ARCHIS, n_train=None, n_val=None,
             n_epochs=None, n_batch=None, T=None, sauver_dataset=True):
    """Entraine un modele et renvoie {archi: chemin_checkpoint}.

    Les arguments n_train ... T valent None par defaut : dans ce cas on garde
    la valeur inscrite dans CFG. Ils ne servent qu'au mode fumee.
    """
    # CFG est un etat global : on le regle avant toute generation de donnees ou
    # construction de modele. Les deux modeles ne different que par les trois
    # champs familles / noise_sweep / dossier.
    CFG.OUT_DIR = conf["dossier"]
    CFG.TRAIN_CMD_RANDOMIZE = True          # meme chemin de code pour A et B
    CFG.TRAIN_CMD_FAMILIES = conf["familles"]
    CFG.TRAIN_NOISE_SWEEP = conf["noise_sweep"]
    CFG.DATASET_PATH = os.path.join(conf["dossier"], "dataset.npz")
    if n_train is not None:
        CFG.N_TRAIN = n_train
    if n_val is not None:
        CFG.N_VAL = n_val
    if n_epochs is not None:
        CFG.N_EPOCHS = n_epochs
    if n_batch is not None:
        CFG.N_BATCH = n_batch
    if T is not None:
        CFG.T = T
    os.makedirs(CFG.OUT_DIR, exist_ok=True)

    barre = "=" * 70
    print(f"\n{barre}\n== Modele {nom} : {conf['commentaire']}\n"
          f"== sortie : {CFG.OUT_DIR}  |  T={CFG.T}  N_TRAIN={CFG.N_TRAIN}  "
          f"N_EPOCHS={CFG.N_EPOCHS}\n{barre}")

    torch.manual_seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    sm = SystemModel()

    # Memes graines pour A et B : seule la distribution des commandes et du
    # bruit change entre les deux jeux de donnees.
    data_train = generate_dataset(sm, CFG.N_TRAIN, seed=CFG.SEED,
                                  noise_sweep=conf["noise_sweep"])
    data_val = generate_dataset(sm, CFG.N_VAL, seed=CFG.SEED + 1,
                                noise_sweep=conf["noise_sweep"])
    data_test = generate_dataset(sm, CFG.N_TEST, seed=CFG.SEED + 99,
                                 noise_sweep=False)
    if sauver_dataset:
        save_dataset(data_train, data_val, data_test)

    chemins = {}
    for archi in archis:
        print(f"\n----- {nom} / {archi} -----")
        # in_mult / out_mult sont des valeurs par defaut evaluees a l'import de
        # KalmanNet_Drones : muter CFG ne les changerait pas. On les passe donc
        # explicitement, pour que tout balayage futur d'architecture fonctionne.
        model = KalmanNetNN(sm, archi=archi,
                            in_mult=CFG.IN_MULT, out_mult=CFG.OUT_MULT)
        ht, hv, ckpt = train(sm, model, data_train, data_val, tag=archi)
        plot_loss(ht, hv, archi, CFG.OUT_DIR)
        chemins[archi] = ckpt
    return chemins


def main(fumee=False):
    if fumee:
        CFG.N_TEST = 2
        print(">> MODE FUMEE : config jouet, les checkpoints produits n'ont "
              "aucune valeur scientifique.")

    tous = {}
    for nom in CONFIGS:
        conf = CONFIGS[nom]
        if fumee:
            tous[nom] = entraine(nom, conf, n_train=8, n_val=4, n_epochs=2,
                                 n_batch=4, T=40, sauver_dataset=False)
        else:
            tous[nom] = entraine(nom, conf)

    print("\n== Checkpoints produits ==")
    for nom in tous:
        for archi in sorted(tous[nom]):
            print(f"   {nom:16s} {archi} -> {tous[nom][archi]}")
    print("\nEtape suivante : python etude_generalisation.py")
    return tous


if __name__ == "__main__":
    main(fumee="--fumee" in sys.argv)
