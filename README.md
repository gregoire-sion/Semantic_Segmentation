"""
Main du filtre centralise a 3 drones.
Genere des trajectoires (commande sinusoidale moyenne nulle), passe l'EKF
baseline, imprime les MSE. KalmanNet vient ensuite (flag RUN_KNET).
"""
import torch
from datetime import datetime

from Simulations.Extended_sysmdl import SystemModel
import Simulations.config as config
from Filters.EKF_test import EKFTest

import Simulations.Drones3.parameters as P
from Simulations.Drones3.parameters import (
    f, h, getJacobian, m, n, dt, Q, R, m1x_0, m2x_0, set_command, I
)

# ----------------------------------------------------------------------
RUN_BASELINE = True     # evaluer l'EKF analytique
RUN_KNET     = False    # passer a True une fois la baseline validee
# ----------------------------------------------------------------------

print("Pipeline Drones3 Start")
strTime = datetime.now().strftime("%m.%d.%y_%H:%M:%S")
print("Current Time =", strTime)

args = config.general_settings()
args.N_E = 200
args.N_CV = 20
args.N_T = 50
args.T = 100
args.T_test = 100
args.randomInit_train = False
args.randomInit_cv = False
args.randomInit_test = False
args.randomLength = False
args.use_cuda = False           # mettre True si GPU dispo

device = torch.device('cuda' if (args.use_cuda and torch.cuda.is_available()) else 'cpu')
print("Using", device)
P.dev = device

#########################################
### Commande sinusoidale (moy. nulle) ###
#########################################
def command_at(t_idx, batch_size):
    """Commande [batch,2,2] au pas t_idx, pour drones 1 et 3. Moyenne nulle."""
    tt = t_idx * dt
    u = torch.zeros(batch_size, 2, 2, device=device)
    # drone 1
    u[:, 0, 0] = 0.8 * torch.sin(torch.tensor(2*torch.pi*0.5*tt))
    u[:, 0, 1] = 0.6 * torch.cos(torch.tensor(2*torch.pi*0.4*tt))
    # drone 3
    u[:, 1, 0] = 0.7 * torch.sin(torch.tensor(2*torch.pi*0.3*tt))
    u[:, 1, 1] = 0.5 * torch.sin(torch.tensor(2*torch.pi*0.6*tt))
    return u

#########################################
### Generateur de trajectoires maison ###
#########################################
def generate(N, T, x0, Q, R, seed=0):
    """
    Genere N trajectoires de longueur T en pilotant set_command a chaque pas.
    Retourne target [N,m,T] (etats vrais) et input [N,n,T] (mesures bruitees).
    """
    torch.manual_seed(seed)
    Lq = torch.linalg.cholesky(Q + 1e-12*torch.eye(m))
    Lr = torch.linalg.cholesky(R + 1e-12*torch.eye(n))
    target = torch.zeros(N, m, T, device=device)
    inp    = torch.zeros(N, n, T, device=device)

    x = x0.reshape(1, m, 1).repeat(N, 1, 1).to(device)   # [N,m,1]
    for t in range(T):
        set_command(command_at(t, N))                    # <-- pilote f()
        x = f(x)                                         # prediction
        wt = (Lq @ torch.randn(N, m, 1, device=device))  # bruit process
        x = x + wt
        target[:, :, t] = x.squeeze(-1)
        yt = h(x)                                        # mesure
        vt = (Lr @ torch.randn(N, n, 1, device=device))  # bruit mesure
        inp[:, :, t] = (yt + vt).squeeze(-1)
    return target, inp

#########################
### System Model      ###
#########################
sys_model = SystemModel(f, Q, h, R, args.T, args.T_test, m, n)
sys_model.InitSequence(m1x_0, m2x_0)

print("Generation des donnees...")
test_target, test_input = generate(args.N_T, args.T_test, m1x_0, Q, R, seed=42)
print("test_target:", test_target.shape, " test_input:", test_input.shape)

#########################
### EKF baseline      ###
#########################
if RUN_BASELINE:
    print("\n=== EKF baseline ===")
    [MSE_arr, MSE_avg, MSE_dB, KG, EKF_out, EKF_sigma] = EKFTest(
        args, sys_model, test_input, test_target, allStates=True)
    print(f"EKF MSE global : {MSE_dB:.2f} dB")
    # MSE par composante d'un drone (ex. position drone 2)
    p2 = I[1]
    for nm_, ci in [("x2", p2['x']), ("vx2", p2['vx']),
                    ("ax2", p2['ax']), ("bx2", p2['bx'])]:
        e = ((EKF_out[:, ci, :] - test_target[:, ci, :])**2).mean()
        print(f"  MSE {nm_:4s}: {10*torch.log10(e):.2f} dB")

#########################
### KalmanNet (option)###
#########################
if RUN_KNET:
    from KNet.KalmanNet_nn import KalmanNetNN
    from Pipelines.Pipeline_EKF import Pipeline_EKF
    print("\n=== KalmanNet ===  (a completer : train/test)")
    # cv/train a generer de la meme facon que test pour comparaison equitable

print("\nFini.")
