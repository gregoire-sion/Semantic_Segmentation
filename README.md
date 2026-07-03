import numpy as np
from KalmanNet_Drones import CFG, SystemModel

sm = SystemModel()
x0 = sm.x0.squeeze(-1).cpu().numpy()

d = np.load(CFG.DATASET_PATH)
print(">> Fichier :", CFG.DATASET_PATH)
print("   cles :", list(d.keys()))

for split in ("Xtr", "Xva", "Xte"):
    X = d[split]
    x_init = X[:, 0, :, 0]
    offs = x_init - x0[None, :]
    std_par_var = offs.std(axis=0)
    print(f"\n== {split} : {X.shape[0]} traj ==")
    for name, b in (("drone1", 0), ("drone2", 8), ("drone3", 16)):
        print(f"  {name} std_offset "
              f"pos=({std_par_var[b]:.2f},{std_par_var[b+1]:.2f}) "
              f"vit=({std_par_var[b+2]:.2f},{std_par_var[b+3]:.2f}) "
              f"acc=({std_par_var[b+4]:.2f},{std_par_var[b+5]:.2f}) "
              f"bias=({std_par_var[b+6]:.2f},{std_par_var[b+7]:.2f})")

print("\n>> Attendu si INIT_OFFSET_P0 actif : std pos~2, vit~0.5, acc~0.5, bias~1")
print(">> Si acc et bias ~0 partout : dataset genere en mode position-only (ancien)")

for split in ("Xtr", "Ytr", "Utr", "Xte"):
    A = d[split]
    print(f"  {split}: shape={A.shape} finite={np.isfinite(A).all()} "
          f"min={A.min():.2f} max={A.max():.2f}")

