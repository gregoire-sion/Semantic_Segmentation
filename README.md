import numpy as np

dt = 0.1
m = 24
n = 8
EPS = 1e-9

# Indices par drone : drone d occupe [8d .. 8d+7] = x,y,vx,vy,ax,ay,bx,by
def idx(d):
    base = 8*d
    return dict(x=base, y=base+1, vx=base+2, vy=base+3,
                ax=base+4, ay=base+5, bx=base+6, by=base+7)

I = [idx(0), idx(1), idx(2)]  # drones 1,2,3

def f(x, u):
    """x: (24,), u: (2,2) commande [ux,uy] pour drones 1 et 3. Retourne x_{t+1}."""
    xn = np.zeros(m)
    for d in range(3):
        p = I[d]
        ax, ay = x[p['ax']], x[p['ay']]
        # cinématique commune (CA sur cet intervalle)
        xn[p['x']]  = x[p['x']]  + x[p['vx']]*dt + 0.5*ax*dt**2
        xn[p['y']]  = x[p['y']]  + x[p['vy']]*dt + 0.5*ay*dt**2
        xn[p['vx']] = x[p['vx']] + ax*dt
        xn[p['vy']] = x[p['vy']] + ay*dt
        if d == 1:  # drone 2 : random walk sur accel + biais
            xn[p['ax']] = ax
            xn[p['ay']] = ay
            xn[p['bx']] = x[p['bx']]
            xn[p['by']] = x[p['by']]
        else:       # drones 1 et 3 : accel = commande, biais gelé
            ui = 0 if d == 0 else 1
            xn[p['ax']] = u[ui,0]
            xn[p['ay']] = u[ui,1]
            xn[p['bx']] = x[p['bx']]
            xn[p['by']] = x[p['by']]
    return xn

def dist(x, di, dj):
    pi, pj = I[di], I[dj]
    dx = x[pi['x']] - x[pj['x']]
    dy = x[pi['y']] - x[pj['y']]
    return np.sqrt(dx*dx + dy*dy)

def h(x):
    """8 mesures : x1,y1, ax2+bx2, ay2+by2, d12, d23, d13, d23(redondant)."""
    p1, p2, p3 = I[0], I[1], I[2]
    y = np.zeros(n)
    y[0] = x[p1['x']]
    y[1] = x[p1['y']]
    y[2] = x[p2['ax']] + x[p2['bx']]
    y[3] = x[p2['ay']] + x[p2['by']]
    y[4] = dist(x, 0, 1)  # d12
    y[5] = dist(x, 1, 2)  # d23
    y[6] = dist(x, 0, 2)  # d13
    y[7] = dist(x, 1, 2)  # d23 redondant
    return y

def H_analytic(x):
    """Jacobienne analytique 8x24 de h."""
    H = np.zeros((n, m))
    p1, p2, p3 = I[0], I[1], I[2]
    # mesures linéaires
    H[0, p1['x']] = 1.0
    H[1, p1['y']] = 1.0
    H[2, p2['ax']] = 1.0; H[2, p2['bx']] = 1.0
    H[3, p2['ay']] = 1.0; H[3, p2['by']] = 1.0
    # distances : ligne row, paire (di,dj)
    def fill_dist(row, di, dj):
        pi, pj = I[di], I[dj]
        dx = x[pi['x']] - x[pj['x']]
        dy = x[pi['y']] - x[pj['y']]
        d = np.sqrt(dx*dx + dy*dy) + EPS
        H[row, pi['x']] =  dx/d; H[row, pi['y']] =  dy/d
        H[row, pj['x']] = -dx/d; H[row, pj['y']] = -dy/d
    fill_dist(4, 0, 1)  # d12
    fill_dist(5, 1, 2)  # d23
    fill_dist(6, 0, 2)  # d13
    fill_dist(7, 1, 2)  # d23 redondant
    return H

# ---- TEST 1 : Jacobienne de h vs différences finies ----
np.random.seed(0)
x0 = np.random.randn(m) * 5  # positions écartées pour éviter d->0
Ha = H_analytic(x0)
Hfd = np.zeros((n, m))
h0 = h(x0)
delta = 1e-6
for k in range(m):
    xp = x0.copy(); xp[k] += delta
    Hfd[:, k] = (h(xp) - h0) / delta
err_h = np.max(np.abs(Ha - Hfd))
print(f"[h] erreur max Jacobienne analytique vs diff finies : {err_h:.3e}")

# ---- TEST 2 : structure (zéros attendus) ----
print(f"[h] mesure 5 et 7 (d23 redondant) identiques : {np.allclose(h0[5], h0[7])}")
print(f"[h] lignes 5 et 7 de H identiques : {np.allclose(Ha[5], Ha[7])}")

# ---- TEST 3 : f produit des trajectoires bornées avec commande sinusoïdale ----
T = 200
x = np.zeros(m)
x[I[0]['x']], x[I[1]['x']], x[I[2]['x']] = 0.0, 10.0, 5.0
x[I[0]['y']], x[I[1]['y']], x[I[2]['y']] = 0.0, 0.0, 8.0
x[I[1]['bx']], x[I[1]['by']] = 0.3, -0.2  # biais vrai du drone 2
traj = np.zeros((T, m))
for t in range(T):
    tt = t*dt
    u = np.array([[0.8*np.sin(2*np.pi*0.5*tt), 0.6*np.cos(2*np.pi*0.4*tt)],
                  [0.7*np.sin(2*np.pi*0.3*tt), 0.5*np.sin(2*np.pi*0.6*tt)]])
    x = f(x, u)
    traj[t] = x
pos_max = np.max(np.abs(traj[:, [0,1,8,9,16,17]]))
print(f"[f] position max sur {T} pas (commande sinus moy. nulle) : {pos_max:.1f}")
print(f"[f] biais drone2 conservé : bx={traj[-1,I[1]['bx']]:.3f}, by={traj[-1,I[1]['by']]:.3f}")
