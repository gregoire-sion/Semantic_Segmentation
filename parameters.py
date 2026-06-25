"""
Parameters for the 3-drone CENTRALIZED KalmanNet model.

================================================================================
SYSTEM DESCRIPTION
================================================================================
Three drones in 2D, fused in a SINGLE centralized filter estimating one
augmented global state.

Per-drone state (8 components), identical layout for all drones (uniformity):
    [ x, y, vx, vy, ax, ay, bx, by ]
Global state (m = 24): [ drone1(8) | drone2(8) | drone3(8) ]

Roles (decided during modelling):
  * Drone 1: acceleration COMMANDED (known input u). Bias = frozen phantom.
  * Drone 2: acceleration UNKNOWN, modelled constant in F (random walk) and
             ESTIMATED. Accelerometer BIASED; bias ESTIMATED (random walk).
  * Drone 3: acceleration COMMANDED (known input u). Bias = frozen phantom.

Dynamics:    x(t+1) = F x(t) + B u(t) + process_noise
Observation: y(t)   = h(x(t)) + measurement_noise

Measurements (n = 8), in the order requested:
    [ x1, y1, ax2+bx2, ay2+by2, d12, d23(by D2), d13, d23(by D3) ]
  - x1, y1              : drone-1 absolute position           (LINEAR)
  - ax2+bx2, ay2+by2    : drone-2 BIASED accelerometer         (LINEAR in state)
  - d12 = ||p1 - p2||   : inter-drone distance (D2)            (NON-LINEAR)
  - d23 = ||p2 - p3||   : inter-drone distance (D2)            (NON-LINEAR)
  - d13 = ||p1 - p3||   : inter-drone distance (D3)            (NON-LINEAR)
  - d23 = ||p2 - p3||   : inter-drone distance (D3), REDUNDANT (NON-LINEAR)

d23 appears twice (measured independently by D2 and D3) with independent noise:
this redundancy improves the relative recalibration between drones 2 and 3.

Observability: with this network of distances (each drone constrained by two
distances) plus drone-1's absolute position anchor, the drone-2 states -- and
crucially its bias -- are observable, PROVIDED the manoeuvre (sine phase of the
command) excites the bias. Verified numerically (rank 22/24; the only
unobservable directions are the frozen drone-1 and drone-3 phantom biases).
================================================================================
"""

import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2

#############################
### Dimensions & indexing ###
#############################
n_drones = 3
per_drone = 8                      # [x, y, vx, vy, ax, ay, bx, by]
m = n_drones * per_drone           # 24
n = 8                              # see measurement list above
dim_u = 2 * n_drones               # [ax1,ay1, ax2,ay2, ax3,ay3] = 6

IDX = {'x':0,'y':1,'vx':2,'vy':3,'ax':4,'ay':5,'bx':6,'by':7}
def gidx(drone, name): return drone*per_drone + IDX[name]

D1, D2, D3 = 0, 1, 2
i_x1,i_y1 = gidx(D1,'x'),gidx(D1,'y')
i_x2,i_y2 = gidx(D2,'x'),gidx(D2,'y')
i_x3,i_y3 = gidx(D3,'x'),gidx(D3,'y')
i_ax2,i_ay2 = gidx(D2,'ax'),gidx(D2,'ay')
i_bx2,i_by2 = gidx(D2,'bx'),gidx(D2,'by')

#############################
### Time step             ###
#############################
delta_t = 0.1   # [s]

#############################################################
### State-evolution matrix F (block-diagonal, 3 x 8x8)    ###
#############################################################
def _single_drone_F(dt):
    F = torch.eye(per_drone)
    F[IDX['x'],  IDX['vx']] = dt
    F[IDX['x'],  IDX['ax']] = 0.5*dt*dt
    F[IDX['y'],  IDX['vy']] = dt
    F[IDX['y'],  IDX['ay']] = 0.5*dt*dt
    F[IDX['vx'], IDX['ax']] = dt
    F[IDX['vy'], IDX['ay']] = dt
    return F

F_block = _single_drone_F(delta_t)
F = torch.zeros(m, m)
for d in range(n_drones):
    s = d*per_drone
    F[s:s+per_drone, s:s+per_drone] = F_block

#############################################################
### Command matrix B                                      ###
#############################################################
B = torch.zeros(m, dim_u)
def _fill_B(B, drone, cax, cay, dt, active):
    if not active: return
    B[gidx(drone,'x'),  cax] = 0.5*dt*dt
    B[gidx(drone,'vx'), cax] = dt
    B[gidx(drone,'ax'), cax] = 1.0
    B[gidx(drone,'y'),  cay] = 0.5*dt*dt
    B[gidx(drone,'vy'), cay] = dt
    B[gidx(drone,'ay'), cay] = 1.0

_fill_B(B, D1, 0, 1, delta_t, active=True)   # drone 1 commanded
_fill_B(B, D2, 2, 3, delta_t, active=False)  # drone 2 NOT commanded
_fill_B(B, D3, 4, 5, delta_t, active=True)   # drone 3 commanded

#############################################################
### Initial state mean and covariance                     ###
#############################################################
m1x_0 = torch.zeros(m, 1)
m1x_0[i_x1,0], m1x_0[i_y1,0] = 0.0, 0.0     # drone 1 at origin (anchor)
m1x_0[i_x2,0], m1x_0[i_y2,0] = 5.0, 0.0     # drone 2
m1x_0[i_x3,0], m1x_0[i_y3,0] = 2.5, 4.0     # drone 3 (forms a triangle)

m2x_0 = torch.zeros(m, m)
m2x_0[i_bx2,i_bx2] = 1.0
m2x_0[i_by2,i_by2] = 1.0

#############################################################
### Process-noise structure Q                             ###
#############################################################
q_weights = torch.zeros(m)
q_weights[i_ax2] = 1.0
q_weights[i_ay2] = 1.0
q_weights[i_bx2] = 1.0
q_weights[i_by2] = 1.0
Q_structure = torch.diag(q_weights)

#############################################################
### Measurement-noise structure R                         ###
#############################################################
r_weights = torch.ones(n)
R_structure = torch.diag(r_weights)

#############################################################
### Observation function h(x) and Jacobian H(x)           ###
#############################################################
EPS = 1e-6
R_X1, R_Y1, R_AX2, R_AY2, R_D12, R_D23a, R_D13, R_D23b = range(n)

def _dist_and_grads(xs, ia_x, ia_y, ib_x, ib_y):
    dx = xs[:, ia_x] - xs[:, ib_x]
    dy = xs[:, ia_y] - xs[:, ib_y]
    d = torch.sqrt(dx*dx + dy*dy + EPS)
    return dx, dy, d

def h(x, jacobian=False):
    b = x.shape[0]
    device = x.device
    xs = x[:, :, 0]
    y = torch.zeros(b, n, 1, device=device)

    y[:, R_X1, 0] = xs[:, i_x1]
    y[:, R_Y1, 0] = xs[:, i_y1]
    y[:, R_AX2, 0] = xs[:, i_ax2] + xs[:, i_bx2]
    y[:, R_AY2, 0] = xs[:, i_ay2] + xs[:, i_by2]

    dx12, dy12, d12 = _dist_and_grads(xs, i_x1, i_y1, i_x2, i_y2)
    dx23, dy23, d23 = _dist_and_grads(xs, i_x2, i_y2, i_x3, i_y3)
    dx13, dy13, d13 = _dist_and_grads(xs, i_x1, i_y1, i_x3, i_y3)
    y[:, R_D12, 0]  = d12
    y[:, R_D23a, 0] = d23
    y[:, R_D13, 0]  = d13
    y[:, R_D23b, 0] = d23

    if not jacobian:
        return y

    H = torch.zeros(b, n, m, device=device)
    H[:, R_X1, i_x1] = 1.0
    H[:, R_Y1, i_y1] = 1.0
    H[:, R_AX2, i_ax2] = 1.0
    H[:, R_AX2, i_bx2] = 1.0
    H[:, R_AY2, i_ay2] = 1.0
    H[:, R_AY2, i_by2] = 1.0

    inv = 1.0/d12
    H[:, R_D12, i_x1] =  dx12*inv; H[:, R_D12, i_x2] = -dx12*inv
    H[:, R_D12, i_y1] =  dy12*inv; H[:, R_D12, i_y2] = -dy12*inv

    inv = 1.0/d23
    H[:, R_D23a, i_x2] =  dx23*inv; H[:, R_D23a, i_x3] = -dx23*inv
    H[:, R_D23a, i_y2] =  dy23*inv; H[:, R_D23a, i_y3] = -dy23*inv

    inv = 1.0/d13
    H[:, R_D13, i_x1] =  dx13*inv; H[:, R_D13, i_x3] = -dx13*inv
    H[:, R_D13, i_y1] =  dy13*inv; H[:, R_D13, i_y3] = -dy13*inv

    inv = 1.0/d23
    H[:, R_D23b, i_x2] =  dx23*inv; H[:, R_D23b, i_x3] = -dx23*inv
    H[:, R_D23b, i_y2] =  dy23*inv; H[:, R_D23b, i_y3] = -dy23*inv

    return y, H


#############################################################
### Base dynamics f(x) and command-aware factory          ###
#############################################################
# The repo always calls the dynamics as a function f(x). We exploit this:
# the commanded dynamics x(t+1) = F x(t) + B u(t) is wrapped INSIDE a function
# f_t(x) = F x + (B u_t), rebuilt at each time step with the current u_t.
#
# Key property: the Jacobian of f_t w.r.t. x is F (the additive term B u_t does
# not depend on x). So the repo's getJacobian(x, f) recovers F automatically,
# and NOTHING else in EKF / KalmanNet needs to change.
#
# Batched convention (matching the repo): x is [batch, m, 1].

def f_linear(x):
    """Autonomous linear dynamics f(x) = F x (no command). Used as a fallback
    and for the Jacobian shape. x: [batch, m, 1] -> [batch, m, 1]."""
    batch = x.shape[0]
    Fb = F.to(x.device).expand(batch, m, m)
    return torch.bmm(Fb, x)

def make_f_commanded(u_t):
    """
    Build the dynamics function for ONE time step, given the command u_t.
      u_t : tensor of shape [dim_u] or [dim_u, 1]  (the command at time t)
    Returns f(x) = F x + B u_t, with the SAME [batch, m, 1] convention.
    Its Jacobian w.r.t. x is F (B u_t is constant in x).
    """
    u_col = u_t.reshape(dim_u, 1)
    Bu = torch.matmul(B, u_col).reshape(1, m, 1)   # [1, m, 1], broadcasts over batch

    def f(x):
        batch = x.shape[0]
        Fb = F.to(x.device).expand(batch, m, m)
        return torch.bmm(Fb, x) + Bu.to(x.device)

    return f


#############################################################
### Command-signal generator u(t): const -> sine -> const ###
#############################################################
def make_command(T, dt=delta_t,
                 freq1=0.25, freq3=0.20,
                 amp_lo=0.3, amp_hi=1.0, frac=(0.33, 0.66)):
    """
    Commanded accelerations for drones 1 and 3 (drone-2 rows stay ZERO).

    IMPORTANT - physical boundedness:
    A *constant* non-zero acceleration makes position grow without bound
    (the drones "take off" to infinity), which breaks the filter. We therefore
    use ZERO-MEAN sinusoidal accelerations at a FIXED frequency per axis, whose
    double integral (position) stays bounded. The "regime change" is encoded as
    a change of AMPLITUDE (gentle -> strong manoeuvre -> gentle), not as
    constant->sine->constant. This keeps trajectories finite while still
    exciting the bias estimation during the strong-manoeuvre middle phase.

    Returns U: [dim_u, T], rows [ax1, ay1, ax2, ay2, ax3, ay3].
    """
    U = torch.zeros(dim_u, T)
    t1, t2 = int(frac[0]*T), int(frac[1]*T)
    t = torch.arange(T).float() * dt
    # amplitude schedule: low -> high -> low
    amp = torch.where(torch.arange(T) < t1, amp_lo,
          torch.where(torch.arange(T) < t2, amp_hi, amp_lo)).float()
    two_pi = 2 * torch.pi
    # Drone 1: x and y in quadrature (90 deg) -> roughly circular manoeuvre
    U[0, :] = amp * torch.sin(two_pi * freq1 * t)
    U[1, :] = amp * torch.sin(two_pi * freq1 * t + torch.pi/2)
    # Drone 3: different frequency and phase -> independent manoeuvre
    U[4, :] = amp * torch.sin(two_pi * freq3 * t + 0.5)
    U[5, :] = amp * torch.sin(two_pi * freq3 * t + 0.5 + torch.pi/2)
    # rows 2,3 (drone 2) stay zero: its acceleration is NOT commanded
    return U


#############################################################
### Drone-2 TRUE acceleration profile (data generation)   ###
#############################################################
def make_true_accel_drone2(T, dt=delta_t,
                           freq=0.22, amp_lo=0.25, amp_hi=0.8, frac=(0.33, 0.66)):
    """
    TRUE (ax2, ay2) used ONLY to build the ground-truth trajectory of drone 2
    (the filter never sees it; it must estimate it). Same bounded, zero-mean
    sinusoidal philosophy as the commands, with an amplitude schedule
    low -> high -> low. The strong middle phase is what makes the accelerometer
    bias observable.

    Returns A2: [2, T] = (ax2, ay2).
    """
    A2 = torch.zeros(2, T)
    t1, t2 = int(frac[0]*T), int(frac[1]*T)
    t = torch.arange(T).float() * dt
    amp = torch.where(torch.arange(T) < t1, amp_lo,
          torch.where(torch.arange(T) < t2, amp_hi, amp_lo)).float()
    two_pi = 2 * torch.pi
    A2[0, :] = amp * torch.sin(two_pi * freq * t + 0.3)
    A2[1, :] = amp * torch.sin(two_pi * freq * t + 0.3 + torch.pi/2)
    return A2


#############################################################
### TRUE accelerometer bias of drone 2 (ground truth)     ###
#############################################################
TRUE_BIAS_DRONE2 = torch.tensor([0.4, -0.3])


if __name__ == "__main__":
    print("m=%d n=%d dim_u=%d"%(m,n,dim_u))
    print("F shape", tuple(F.shape), "B shape", tuple(B.shape))
    ok = True
    for a in range(n_drones):
        for b_ in range(n_drones):
            if a != b_:
                blk = F[a*per_drone:(a+1)*per_drone, b_*per_drone:(b_+1)*per_drone]
                ok = ok and bool(torch.all(blk == 0))
    print("F block-diagonal:", ok)
    print("Drone-2 NOT commanded (B cols 2,3 zero):", bool(torch.all(B[:,2:4]==0)))
    print("Drone-1/3 commanded:", bool(B[gidx(D1,'ax'),0]==1.0) and bool(B[gidx(D3,'ax'),4]==1.0))
    print("Q only drone-2 accel+bias active:",
          float(Q_structure[i_ax2,i_ax2])==1.0 and
          float(Q_structure[gidx(D1,'bx'),gidx(D1,'bx')])==0.0)
    x = m1x_0.reshape(1,m,1).repeat(3,1,1)
    y, H = h(x, jacobian=True)
    print("y shape", tuple(y.shape), "H shape", tuple(H.shape))
    print("Init distances [d12,d23,d13,d23]:",
          [round(float(y[0,k,0]),3) for k in (R_D12,R_D23a,R_D13,R_D23b)])
