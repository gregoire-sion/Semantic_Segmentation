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
### Commande courante     ###
#############################
# u_current : tenseur [batch, 2, 2] = commande [ux,uy] pour drones 1 et 3.
# Mis a jour par le generateur a chaque pas (set_command), lu par f().
# Par defaut nul (sera ecrase). On le garde en variable de module pour
# respecter la signature f(x) imposee par Extended_sysmdl.
u_current = None

def set_command(u):
    """u : [batch, 2, 2] -> commande du pas courant pour drones 1 et 3."""
    global u_current
    u_current = u

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

    # commande du pas courant (zeros si non definie)
    if u_current is None:
        u = torch.zeros(bs, 2, 2, device=x.device, dtype=x.dtype)
    else:
        u = u_current.to(x.device).type_as(x)

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
