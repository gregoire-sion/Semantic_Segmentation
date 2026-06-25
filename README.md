"""
Parametres du filtre centralise a 3 drones (etat augmente, mesures non-lineaires).

Etat (m=24) : 3 drones x 8 composantes [x, y, vx, vy, ax, ay, bx, by]
  - Drone 1 (cols  0-7 ) : accel commandee (a = u), biais gele
  - Drone 2 (cols  8-15) : accel inconnue (random walk) + biais estime
  - Drone 3 (cols 16-23) : accel commandee (a = u), biais gele

Mesures (n=8) : x1, y1, ax2+bx2, ay2+by2, d12, d23, d13, d23(redondant)

Convention repo : f et h prennent x de forme [batch, m, 1] (ou [batch, n, 1]),
et acceptent jacobian=True pour renvoyer aussi la Jacobienne batchee.
"""
import torch

#############################
### Dimensions & timing   ###
#############################
m = 24            # dimension etat
n = 8             # dimension mesure
dt = 0.1          # pas de temps [s]
EPS = 1e-9        # garde-fou singularite distance

# device par defaut (surcharge possible depuis le main)
dev = torch.device('cpu')

#############################
### Indices par drone     ###
#############################
# drone d occupe les colonnes [8d .. 8d+7]
def _idx(d):
    b = 8 * d
    return dict(x=b, y=b+1, vx=b+2, vy=b+3, ax=b+4, ay=b+5, bx=b+6, by=b+7)
I = [_idx(0), _idx(1), _idx(2)]

#############################
### Commande (sequence)   ###
#############################
# La commande u(t) des drones 1 et 3 est DETERMINISTE et CONNUE du filtre.
# On la stocke comme une sequence indexee par le temps : _u_seq[t] = [2,2].
# f() lit la commande du pas courant via _u_ptr et avance le pointeur.
# Ainsi le generateur ET l'EKF rejouent EXACTEMENT la meme u(t) :
#   - reset_command(): remet le pointeur a 0 (a appeler avant chaque passe)
#   - set_command_sequence(useq): fixe la sequence complete [T,2,2]
_u_seq = None      # tenseur [T, 2, 2] (commande par pas, identique pour tout le batch)
_u_ptr = 0         # pointeur de pas courant

def set_command_sequence(useq):
    """useq : [T, 2, 2] -> sequence de commandes pour drones 1 et 3."""
    global _u_seq, _u_ptr
    _u_seq = useq
    _u_ptr = 0

def reset_command():
    """Remet le pointeur de commande a 0 (debut de sequence)."""
    global _u_ptr
    _u_ptr = 0

def _next_command(batch_size, device, dtype):
    """Renvoie la commande [batch,2,2] du pas courant et avance le pointeur."""
    global _u_ptr
    if _u_seq is None:
        return torch.zeros(batch_size, 2, 2, device=device, dtype=dtype)
    t = min(_u_ptr, _u_seq.shape[0] - 1)   # securite si depassement
    u_t = _u_seq[t].to(device=device, dtype=dtype)        # [2,2]
    _u_ptr += 1
    return u_t.unsqueeze(0).expand(batch_size, -1, -1)     # [batch,2,2]

#############################
### Dynamique f(x)        ###
#############################
def f(x, jacobian=False):
    """
    x : [batch, m, 1]
    retourne x_next [batch, m, 1] ; si jacobian=True, retourne (x_next, F[batch,m,m]).
    """
    bs = x.shape[0]
    xs = x.squeeze(-1)                      # [batch, m]
    xn = torch.zeros_like(xs)               # [batch, m]

    # commande du pas courant (lue dans la sequence, avance le pointeur)
    u = _next_command(bs, x.device, x.dtype)   # [batch, 2, 2]

    for d in range(3):
        p = I[d]
        ax = xs[:, p['ax']]
        ay = xs[:, p['ay']]
        # cinematique commune (CA sur l'intervalle dt)
        xn[:, p['x']]  = xs[:, p['x']]  + xs[:, p['vx']]*dt + 0.5*ax*dt**2
        xn[:, p['y']]  = xs[:, p['y']]  + xs[:, p['vy']]*dt + 0.5*ay*dt**2
        xn[:, p['vx']] = xs[:, p['vx']] + ax*dt
        xn[:, p['vy']] = xs[:, p['vy']] + ay*dt
        if d == 1:  # drone 2 : random walk accel + biais
            xn[:, p['ax']] = ax
            xn[:, p['ay']] = ay
            xn[:, p['bx']] = xs[:, p['bx']]
            xn[:, p['by']] = xs[:, p['by']]
        else:       # drones 1 et 3 : accel = commande, biais gele
            ui = 0 if d == 0 else 1
            xn[:, p['ax']] = u[:, ui, 0]
            xn[:, p['ay']] = u[:, ui, 1]
            xn[:, p['bx']] = xs[:, p['bx']]
            xn[:, p['by']] = xs[:, p['by']]

    x_next = xn.unsqueeze(-1)               # [batch, m, 1]

    if not jacobian:
        return x_next

    # Jacobienne F = d f / d x : identique pour tout le batch (dynamique affine)
    F = torch.zeros(m, m, device=x.device, dtype=x.dtype)
    for d in range(3):
        p = I[d]
        # position depend de pos, vit, accel
        F[p['x'],  p['x']]  = 1.0
        F[p['x'],  p['vx']] = dt
        F[p['x'],  p['ax']] = 0.5*dt**2
        F[p['y'],  p['y']]  = 1.0
        F[p['y'],  p['vy']] = dt
        F[p['y'],  p['ay']] = 0.5*dt**2
        # vitesse depend de vit, accel
        F[p['vx'], p['vx']] = 1.0
        F[p['vx'], p['ax']] = dt
        F[p['vy'], p['vy']] = 1.0
        F[p['vy'], p['ay']] = dt
        # biais gele/random walk : b_next = b
        F[p['bx'], p['bx']] = 1.0
        F[p['by'], p['by']] = 1.0
        if d == 1:  # drone 2 : accel = random walk -> d ax_next/d ax = 1
            F[p['ax'], p['ax']] = 1.0
            F[p['ay'], p['ay']] = 1.0
        # drones 1 & 3 : accel = commande (independante de l'etat) -> lignes accel nulles
    F = F.unsqueeze(0).repeat(bs, 1, 1)     # [batch, m, m]
    return x_next, F

#############################
### Mesure h(x)           ###
#############################
def _dist(xs, di, dj):
    pi, pj = I[di], I[dj]
    dx = xs[:, pi['x']] - xs[:, pj['x']]
    dy = xs[:, pi['y']] - xs[:, pj['y']]
    d = torch.sqrt(dx*dx + dy*dy + EPS)
    return d, dx, dy

def h(x, jacobian=False):
    """
    x : [batch, m, 1]
    retourne y [batch, n, 1] ; si jacobian=True, retourne (y, H[batch,n,m]).
    """
    bs = x.shape[0]
    xs = x.squeeze(-1)
    y = torch.zeros(bs, n, device=x.device, dtype=x.dtype)

    p1, p2, p3 = I[0], I[1], I[2]
    y[:, 0] = xs[:, p1['x']]
    y[:, 1] = xs[:, p1['y']]
    y[:, 2] = xs[:, p2['ax']] + xs[:, p2['bx']]
    y[:, 3] = xs[:, p2['ay']] + xs[:, p2['by']]
    d12, dx12, dy12 = _dist(xs, 0, 1); y[:, 4] = d12
    d23, dx23, dy23 = _dist(xs, 1, 2); y[:, 5] = d23
    d13, dx13, dy13 = _dist(xs, 0, 2); y[:, 6] = d13
    y[:, 7] = d23                                  # d23 redondant

    y_out = y.unsqueeze(-1)
    if not jacobian:
        return y_out

    H = torch.zeros(bs, n, m, device=x.device, dtype=x.dtype)
    # mesures lineaires
    H[:, 0, p1['x']] = 1.0
    H[:, 1, p1['y']] = 1.0
    H[:, 2, p2['ax']] = 1.0; H[:, 2, p2['bx']] = 1.0
    H[:, 3, p2['ay']] = 1.0; H[:, 3, p2['by']] = 1.0

    def fill(row, di, dj, d, dx, dy):
        pi, pj = I[di], I[dj]
        H[:, row, pi['x']] =  dx/d; H[:, row, pi['y']] =  dy/d
        H[:, row, pj['x']] = -dx/d; H[:, row, pj['y']] = -dy/d
    fill(4, 0, 1, d12, dx12, dy12)
    fill(5, 1, 2, d23, dx23, dy23)
    fill(6, 0, 2, d13, dx13, dy13)
    fill(7, 1, 2, d23, dx23, dy23)                  # ligne identique a 5

    return y_out, H

#############################
### Bruits Q et R         ###
#############################
def build_Q(q_pos=1e-4, q_vel=1e-3, q_acc2=1e-2, q_bias2=1e-5):
    """
    Q diagonal 24x24 selon les roles :
      drones 1&3 : bruit faible pos/vit, ZERO sur accel (commande det.), ZERO sur biais (gele)
      drone 2    : bruit pos/vit, q_acc2 sur accel (random walk), q_bias2 sur biais (rw lent)
    """
    qd = torch.zeros(m)
    for d in range(3):
        p = I[d]
        qd[p['x']] = q_pos;  qd[p['y']] = q_pos
        qd[p['vx']] = q_vel; qd[p['vy']] = q_vel
        if d == 1:
            qd[p['ax']] = q_acc2;  qd[p['ay']] = q_acc2
            qd[p['bx']] = q_bias2; qd[p['by']] = q_bias2
        # drones 1&3 : accel et biais a 0 (deja zero)
    return torch.diag(qd)

def build_R(r_pos=1e-2, r_acc=1e-2, r_dist=5e-2, r_dist_redund=1e-1):
    """
    R diagonal 8x8. La mesure 7 (d23 redondante) a son propre bruit (capteur drone 3).
    """
    rd = torch.tensor([
        r_pos, r_pos,            # x1, y1
        r_acc, r_acc,            # ax2+bx2, ay2+by2
        r_dist,                  # d12
        r_dist,                  # d23 (capteur drone 2)
        r_dist,                  # d13
        r_dist_redund,           # d23 (capteur drone 3, redondant)
    ])
    return torch.diag(rd)

Q = build_Q()
R = build_R()

#############################
### getJacobian (API repo)###
#############################
def getJacobian(x, g):
    """
    x : [batch, m/n, 1] ; g : fonction f ou h supportant jacobian=True.
    retourne la Jacobienne batchee.
    """
    _, Jac = g(x, jacobian=True)
    return Jac

#############################
### Conditions initiales  ###
#############################
def initial_state():
    """Etat initial nominal [m,1] : positions ecartees, vit/accel nulles, biais drone2 non nul."""
    x0 = torch.zeros(m, 1)
    x0[I[0]['x'], 0], x0[I[0]['y'], 0] = 0.0, 0.0
    x0[I[1]['x'], 0], x0[I[1]['y'], 0] = 10.0, 0.0
    x0[I[2]['x'], 0], x0[I[2]['y'], 0] = 5.0, 8.0
    x0[I[1]['bx'], 0], x0[I[1]['by'], 0] = 0.3, -0.2   # biais vrai drone 2
    return x0

m1x_0 = initial_state()
m2x_0 = torch.eye(m) * 1.0      # P0 (a regler selon confiance initiale)

#############################
### Sequence de commande  ###
#############################
def build_command_sequence(T, device=None, dtype=torch.float32):
    """
    Construit la sequence DETERMINISTE u(t) [T,2,2] : commande sinusoidale
    moyenne nulle pour drones 1 (indice 0) et 3 (indice 1).
    C'est CETTE fonction qui doit etre utilisee a la fois par le generateur
    et par l'EKF, pour garantir une commande identique.
    """
    if device is None:
        device = dev
    t = torch.arange(T, device=device, dtype=dtype) * dt   # [T]
    two_pi = 2 * torch.pi
    useq = torch.zeros(T, 2, 2, device=device, dtype=dtype)
    # drone 1
    useq[:, 0, 0] = 0.8 * torch.sin(two_pi * 0.5 * t)
    useq[:, 0, 1] = 0.6 * torch.cos(two_pi * 0.4 * t)
    # drone 3
    useq[:, 1, 0] = 0.7 * torch.sin(two_pi * 0.3 * t)
    useq[:, 1, 1] = 0.5 * torch.sin(two_pi * 0.6 * t)
    return useq

"""# **Class: Extended Kalman Filter**
Theoretical Non Linear Kalman
"""
import torch

from Simulations.Lorenz_Atractor.parameters import getJacobian

class ExtendedKalmanFilter:

    def __init__(self, SystemModel, args):
        # Device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # process model
        self.f = SystemModel.f
        self.m = SystemModel.m
        self.Q = SystemModel.Q.to(self.device)
        # observation model
        self.h = SystemModel.h
        self.n = SystemModel.n
        self.R = SystemModel.R.to(self.device)
        # sequence length (use maximum length if random length case)
        self.T = SystemModel.T
        self.T_test = SystemModel.T_test
  
    # Predict
    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.m1x_posterior).to(self.device)
        # Compute the Jacobians
        self.UpdateJacobians(getJacobian(self.m1x_posterior,self.f), getJacobian(self.m1x_prior, self.h))
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.bmm(self.batched_F, self.m2x_posterior)
        self.m2x_prior = torch.bmm(self.m2x_prior, self.batched_F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)
        # Predict the 2-nd moment of y
        self.m2y = torch.bmm(self.batched_H, self.m2x_prior)
        self.m2y = torch.bmm(self.m2y, self.batched_H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.bmm(self.m2x_prior, self.batched_H_T)
        self.KG = torch.bmm(self.KG, torch.inverse(self.m2y))

        #Save KalmanGain
        self.KG_array[:,:,:,self.i] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.bmm(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.bmm(self.m2y, torch.transpose(self.KG, 1, 2))
        self.m2x_posterior = self.m2x_prior - torch.bmm(self.KG, self.m2x_posterior)

    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior

    #########################

    def UpdateJacobians(self, F, H):
        self.batched_F = F.to(self.device)
        self.batched_F_T = torch.transpose(F,1,2)
        self.batched_H = H.to(self.device)
        self.batched_H_T = torch.transpose(H,1,2)
    
    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):

            self.m1x_0_batch = m1x_0_batch # [batch_size, m, 1]
            self.m2x_0_batch = m2x_0_batch # [batch_size, m, m]

    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, y):
        """
        input y: batch of observations [batch_size, n, T]
        """
        y = y.to(self.device)
        self.batch_size = y.shape[0] # batch size
        T = y.shape[2] # sequence length (maximum length if randomLength=True)

        # Pre allocate KG array
        self.KG_array = torch.zeros([self.batch_size,self.m,self.n,T]).to(self.device)
        self.i = 0 # Index for KG_array alocation

        # Allocate Array for 1st and 2nd order moments (use zero padding)
        self.x = torch.zeros(self.batch_size, self.m, T).to(self.device)
        self.sigma = torch.zeros(self.batch_size, self.m, self.m, T).to(self.device)
            
        # Set 1st and 2nd order moments for t=0
        self.m1x_posterior = self.m1x_0_batch.to(self.device)
        self.m2x_posterior = self.m2x_0_batch.to(self.device)

        # --- Commande deterministe : rejouer la MEME u(t) que le generateur ---
        # On installe la sequence et on remet le pointeur a 0, de sorte que le
        # f(x) interne de Predict() lise la bonne commande a chaque pas.
        from Simulations.Drones3 import parameters as P
        P.set_command_sequence(P.build_command_sequence(T, device=self.device))
        P.reset_command()

        # Generate in a batched manner
        for t in range(0, T):
            yt = torch.unsqueeze(y[:, :, t],2)
            xt,sigmat = self.Update(yt)
            self.x[:, :, t] = torch.squeeze(xt,2)
            self.sigma[:, :, :, t] = sigmat

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
    f, h, getJacobian, m, n, dt, Q, R, m1x_0, m2x_0, I,
    set_command_sequence, reset_command, build_command_sequence
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
### Generateur de trajectoires maison ###
#########################################
def generate(N, T, x0, Q, R, seed=0):
    """
    Genere N trajectoires de longueur T en rejouant la sequence de commande
    DETERMINISTE (la meme que l'EKF). Retourne target [N,m,T] et input [N,n,T].
    """
    torch.manual_seed(seed)
    Lq = torch.linalg.cholesky(Q + 1e-12*torch.eye(m))
    Lr = torch.linalg.cholesky(R + 1e-12*torch.eye(n))
    target = torch.zeros(N, m, T, device=device)
    inp    = torch.zeros(N, n, T, device=device)

    # installer la sequence de commande partagee et remettre le pointeur a 0
    set_command_sequence(build_command_sequence(T, device=device))
    reset_command()

    x = x0.reshape(1, m, 1).repeat(N, 1, 1).to(device)   # [N,m,1]
    for t in range(T):
        x = f(x)                                         # prediction (lit u(t))
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

