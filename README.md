"""
Parameters for a 3-drone CENTRALIZED collaborative-navigation example,
written to match the KalmanNet_TSP batched conventions:

  * f(x, jacobian=False): x is [batch, m, 1] -> [batch, m, 1]
    when jacobian=True also returns F of shape [batch, m, m]
  * h(x, jacobian=False): x is [batch, m, 1] -> [batch, n, 1]
    when jacobian=True also returns H of shape [batch, n, m]
  * getJacobian(x, g) just calls g(x, jacobian=True) and returns the matrix.

JOINT STATE  (m = 14), 2-D plane (x, y):
  D1 (Constant Velocity):  x1, vx1, y1, vy1            indices 0..3
  D2 (Constant Accel.)  :  x2, vx2, ax2, y2, vy2, ay2  indices 4..9
  D3 (Constant Velocity):  x3, vx3, y3, vy3            indices 10..13

OBSERVATION  (n = 8):
  [ x1,            # GPS of D1 (x)
    y1,            # GPS of D1 (y)
    d12,           # D2 measures range to D1
    d23_byD2,      # D2 measures range to D3
    ax2_meas,      # D2 measures its own accel x  (noisy / biased -> big R)
    ay2_meas,      # D2 measures its own accel y  (noisy / biased -> big R)
    d13,           # D3 measures range to D1
    d23_byD3 ]     # D3 measures range to D2

Drones D2 and D3 have NO absolute position measurement: they are observable
only through their coupling (ranges) to D1, which is GPS-anchored. Check
observability with the EKF before training (see notes in the chat).

Control input for D1 / D3 is folded into the process noise Q here (see chat
for the explicit-control variant).
"""

import torch
import math

#############################
### Dimensions            ###
#############################
m = 14   # joint state
n = 8    # observations

delta_t = 0.1

#############################
### State transition F    ###
#############################
def _cv_block(dt):
    # [pos, vel] -> [[1, dt],[0,1]]
    return torch.tensor([[1.0, dt],
                         [0.0, 1.0]]).float()

def _ca_block(dt):
    # [pos, vel, acc] -> [[1,dt,0.5dt^2],[0,1,dt],[0,0,1]]
    return torch.tensor([[1.0, dt, 0.5*dt*dt],
                         [0.0, 1.0, dt],
                         [0.0, 0.0, 1.0]]).float()

F_const = torch.zeros((m, m)).float()
# D1: two CV axes  x:(0,1)  y:(2,3)
F_const[0:2, 0:2] = _cv_block(delta_t)
F_const[2:4, 2:4] = _cv_block(delta_t)
# D2: two CA axes  x:(4,5,6)  y:(7,8,9)
F_const[4:7, 4:7] = _ca_block(delta_t)
F_const[7:10, 7:10] = _ca_block(delta_t)
# D3: two CV axes  x:(10,11) y:(12,13)
F_const[10:12, 10:12] = _cv_block(delta_t)
F_const[12:14, 12:14] = _cv_block(delta_t)


def f(x, jacobian=False):
    """
    x: [batch, m, 1]
    F is constant (linear motion) -> same F for every batch element.
    """
    batch = x.shape[0]
    F = F_const.to(x.device).reshape(1, m, m).repeat(batch, 1, 1)  # [batch,m,m]
    xt = torch.bmm(F, x)                                           # [batch,m,1]
    if jacobian:
        return xt, F
    return xt


#############################
### Observation h         ###
#############################
def _safe_norm(dx, dy):
    return torch.sqrt(dx*dx + dy*dy + 1e-9)


def h(x, jacobian=False):
    """
    x: [batch, m, 1]  ->  y: [batch, n, 1]
    Non-linear because of the inter-drone ranges d_ij = ||p_i - p_j||.
    """
    batch = x.shape[0]
    xs = x[:, :, 0]  # [batch, m] convenience view

    x1 = xs[:, 0];  y1 = xs[:, 2]
    x2 = xs[:, 4];  y2 = xs[:, 7]
    ax2 = xs[:, 6]; ay2 = xs[:, 9]
    x3 = xs[:, 10]; y3 = xs[:, 12]

    d12 = _safe_norm(x1 - x2, y1 - y2)
    d23 = _safe_norm(x2 - x3, y2 - y3)
    d13 = _safe_norm(x1 - x3, y1 - y3)

    y = torch.stack([x1, y1, d12, d23, ax2, ay2, d13, d23], dim=1)  # [batch, n]
    y = y.reshape(batch, n, 1)

    if jacobian:
        H = _H_jacobian(xs)   # [batch, n, m]
        return y, H
    return y


def _H_jacobian(xs):
    """
    Analytic Jacobian of h w.r.t. the full state.
    xs: [batch, m]   ->   H: [batch, n, m]
    Row order matches h():
      0: x1   -> d/dx1 = 1
      1: y1   -> d/dy1 = 1
      2: d12  -> depends on (x1,y1,x2,y2)
      3: d23  -> depends on (x2,y2,x3,y3)
      4: ax2  -> d/dax2 = 1
      5: ay2  -> d/day2 = 1
      6: d13  -> depends on (x1,y1,x3,y3)
      7: d23  -> depends on (x2,y2,x3,y3)
    """
    batch = xs.shape[0]
    H = torch.zeros(batch, n, m, device=xs.device)

    x1 = xs[:, 0];  y1 = xs[:, 2]
    x2 = xs[:, 4];  y2 = xs[:, 7]
    x3 = xs[:, 10]; y3 = xs[:, 12]

    # state indices
    iX1, iY1 = 0, 2
    iX2, iY2 = 4, 7
    iAX2, iAY2 = 6, 9
    iX3, iY3 = 10, 12

    # Row 0: x1
    H[:, 0, iX1] = 1.0
    # Row 1: y1
    H[:, 1, iY1] = 1.0

    # Row 2: d12 = ||p1 - p2||
    d12 = _safe_norm(x1 - x2, y1 - y2)
    H[:, 2, iX1] = (x1 - x2) / d12
    H[:, 2, iY1] = (y1 - y2) / d12
    H[:, 2, iX2] = (x2 - x1) / d12
    H[:, 2, iY2] = (y2 - y1) / d12

    # Row 3: d23 = ||p2 - p3||  (measured by D2)
    d23 = _safe_norm(x2 - x3, y2 - y3)
    H[:, 3, iX2] = (x2 - x3) / d23
    H[:, 3, iY2] = (y2 - y3) / d23
    H[:, 3, iX3] = (x3 - x2) / d23
    H[:, 3, iY3] = (y3 - y2) / d23

    # Row 4: ax2
    H[:, 4, iAX2] = 1.0
    # Row 5: ay2
    H[:, 5, iAY2] = 1.0

    # Row 6: d13 = ||p1 - p3||
    d13 = _safe_norm(x1 - x3, y1 - y3)
    H[:, 6, iX1] = (x1 - x3) / d13
    H[:, 6, iY1] = (y1 - y3) / d13
    H[:, 6, iX3] = (x3 - x1) / d13
    H[:, 6, iY3] = (y3 - y1) / d13

    # Row 7: d23 again (measured by D3) -> same derivatives as row 3
    H[:, 7, iX2] = (x2 - x3) / d23
    H[:, 7, iY2] = (y2 - y3) / d23
    H[:, 7, iX3] = (x3 - x2) / d23
    H[:, 7, iY3] = (y3 - y2) / d23

    return H


#############################
### getJacobian (repo API) ##
#############################
def getJacobian(x, g):
    _, Jac = g(x, jacobian=True)
    return Jac


#############################
### Q and R structures     ##
#############################
# Process noise: low everywhere, a bit larger on D2's accelerations
# (random-walk on accel) and on the velocities of D1/D3 (absorbs the
# control input we are NOT injecting explicitly).
Q_structure = torch.eye(m).float()
Q_structure[1, 1]   = 5.0   # vx1  (control absorbed here)
Q_structure[3, 3]   = 5.0   # vy1
Q_structure[6, 6]   = 5.0   # ax2  random walk
Q_structure[9, 9]   = 5.0   # ay2
Q_structure[11, 11] = 5.0   # vx3  (control absorbed here)
Q_structure[13, 13] = 5.0   # vy3

# Observation noise: GPS rows tight, range rows moderate,
# D2 accel rows (4,5) DELIBERATELY large -> the "biased/degraded" sensor.
R_structure = torch.eye(n).float()
R_structure[0, 0] = 1.0     # x1 GPS
R_structure[1, 1] = 1.0     # y1 GPS
R_structure[2, 2] = 1.0     # d12
R_structure[3, 3] = 1.0     # d23 (D2)
R_structure[4, 4] = 50.0    # ax2 measured  <-- big = biased/degraded
R_structure[5, 5] = 50.0    # ay2 measured  <-- big
R_structure[6, 6] = 1.0     # d13
R_structure[7, 7] = 1.0     # d23 (D3)


#############################
### Initial state          ##
#############################
# A simple non-degenerate triangle so ranges are well-conditioned at t=0.
# (Avoid colinear drones -> keep the joint state observable.)
m1x_0 = torch.zeros(m, 1).float()
m1x_0[0, 0]  = 0.0;   m1x_0[2, 0]  = 0.0     # D1 at (0,0)
m1x_0[4, 0]  = 10.0;  m1x_0[7, 0]  = 0.0     # D2 at (10,0)
m1x_0[10, 0] = 5.0;   m1x_0[12, 0] = 8.0     # D3 at (5,8)
# small initial velocities so the formation actually moves
m1x_0[1, 0]  = 1.0;   m1x_0[3, 0]  = 0.5     # D1 velocity
m1x_0[11, 0] = 0.8;   m1x_0[13, 0] = 0.6     # D3 velocity

m2x_0 = 1.0 * torch.eye(m).float()           # initial covariance P0
