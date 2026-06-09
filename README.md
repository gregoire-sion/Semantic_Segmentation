import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance 
from numpy.linalg import inv
from scipy.linalg import block_diag

t_max = 16
dt = 0.1
dt_capteur = 16
n_drone = 3
n_variable_etat = 8
n_mesures = 7

temps = [0]
temps_capteur = []

sigma_gps_1 = 0.1
sigma_acc_2 = 0.5
sigma_dist_12 = 0.1
sigma_dist_23 = 0.1
sigma_dist_13 = 0.1

sigma_x_1 = 0.000001
sigma_y_1 = 0.000001
sigma_vx_1 = 0.000001
sigma_vy_1 = 0.000001
sigma_ax_1 = 0.1
sigma_ay_1 = 0.1
sigma_bx_1 = 0.000001
sigma_by_1 = 0.000001

sigma_x_2 = 0.000001
sigma_y_2 = 0.000001
sigma_vx_2 = 0.000001
sigma_vy_2 = 0.000001
sigma_ax_2 = 0.1
sigma_ay_2= 0.1
sigma_bx_2 = 0.000001
sigma_by_2 = 0.000001

sigma_x_3 = 0.000001
sigma_y_3 = 0.000001
sigma_vx_3 = 0.000001
sigma_vy_3 = 0.000001
sigma_ax_3 = 0.1
sigma_ay_3 = 0.1
sigma_bx_3 = 0.000001
sigma_by_3 = 0.000001

omega_x = 0.00001 #rad/s
omega_y = 0.000001

Ax1 = 0.0
theta_0_1x = 0.0
Ay1 = 2.0
theta_0_1y = 0.0
Ax2 = 0.0
theta_0_2x = 0.0
Ay2 = 2.0
theta_0_2y = 0.0
Ax3 = 0.0
theta_0_3x = 0.0
Ay3 = 2.0
theta_0_3y = 0.0

phi_x = 0.0
phi_y = 0.0

################################
#------------Vrai---------------
################################

traj_vrai = np.zeros((int(t_max/dt), n_variable_etat * n_drone))
traj_kalman = np.zeros((int(t_max/dt), n_variable_etat * n_drone))
mesures_capteur = np.zeros((int(t_max/dt_capteur), n_mesures)) # à voir comment je vais m'organiser

X_vrai_1 = np.array([0, 10, 1.0, 0.0, 0, 0, 0, 0])
X_vrai_2 = np.array([10, 0, 1.0, 0.0, 0, 0, 0.5, -0.2])
X_vrai_3 = np.array([0, -10, 1.0, 0.0, 0, 0, 0, 0])

X_vrai = np.concatenate((X_vrai_1, X_vrai_2, X_vrai_3))
print(f"X_vrai décalé aléatoirement : {X_vrai}")

u_vrai = np.array([Ax1*np.cos(phi_x), Ay1*np.sin(phi_y), Ax2*np.cos(phi_x), Ay2*np.sin(phi_y), Ax3*np.cos(phi_x), Ay3*np.sin(phi_y)])

u_vrai_x1_list = []
u_vrai_y1_list = []

B1 = np.array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],])

B2 = np.array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],])

B3 = np.array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],])

B_vrai = np.concatenate((B1, B2, B3), axis=0)

F_vrai_1 = np.array([
    [1, 0, dt, 0, 0.5*dt*dt, 0, 0, 0],
    [0, 1, 0, dt, 0, 0.5*dt*dt, 0, 0],
    [0, 0, 1, 0, dt, 0, 0, 0],
    [0, 0, 0, 1, 0, dt, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]])

F_vrai_2 = np.array([
    [1, 0, dt, 0, 0.5*dt*dt, 0, 0, 0],
    [0, 1, 0, dt, 0, 0.5*dt*dt, 0, 0],
    [0, 0, 1, 0, dt, 0, 0, 0],
    [0, 0, 0, 1, 0, dt, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]])

F_vrai_3 = np.array([
    [1, 0, dt, 0, 0.5*dt*dt, 0, 0, 0],
    [0, 1, 0, dt, 0, 0.5*dt*dt, 0, 0],
    [0, 0, 1, 0, dt, 0, 0, 0],
    [0, 0, 0, 1, 0, dt, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]])

F_vrai = block_diag(F_vrai_1, F_vrai_2, F_vrai_3)

################################
#-----------Capteur-------------
################################

derive_x = 0.001
derive_y = 0.0

d12_init = np.sqrt((X_vrai[0]-X_vrai[8])**2 + (X_vrai[1]-X_vrai[9])**2)
d23_init = np.sqrt((X_vrai[8]-X_vrai[16])**2 + (X_vrai[9]-X_vrai[17])**2)
d13_init = np.sqrt((X_vrai[0]-X_vrai[16])**2 + (X_vrai[2]-X_vrai[17])**2)

X_capteur = np.array([X_vrai[0], X_vrai[1], X_vrai[12] + X_vrai[14], X_vrai[13] + X_vrai[15], d12_init, d23_init, d13_init])

################################
#-------------Kalman------------
################################

sigma_x_init = 0.001
sigma_v_init = 0.001
sigma_a_init = 0.001
sigma_b_init = 0.001

erreur_init = np.array([np.random.normal(0, sigma_x_init), np.random.normal(0, sigma_x_init), np.random.normal(0, sigma_v_init), np.random.normal(0, sigma_v_init), np.random.normal(0, sigma_a_init), np.random.normal(0, sigma_a_init), np.random.normal(0, sigma_b_init), np.random.normal(0, sigma_b_init), np.random.normal(0, sigma_x_init), np.random.normal(0, sigma_x_init), np.random.normal(0, sigma_v_init), np.random.normal(0, sigma_v_init), np.random.normal(0, sigma_a_init), np.random.normal(0, sigma_a_init), np.random.normal(0, sigma_b_init), np.random.normal(0, sigma_b_init), np.random.normal(0, sigma_x_init), np.random.normal(0, sigma_x_init), np.random.normal(0, sigma_v_init), np.random.normal(0, sigma_v_init), np.random.normal(0, sigma_a_init), np.random.normal(0, sigma_a_init), np.random.normal(0, sigma_b_init), np.random.normal(0, sigma_b_init)])


X_est = X_vrai.copy() + erreur_init
X_est[6:8] = [0.0,0.0]
X_est[14:16] = [0.0,0.0]
X_est[22:24] = [0.0,0.0]

u_kalman = u_vrai.copy()

print(f"Vecteur initial vrai : {X_vrai}")
print(f"Vecteur initial décalé : {X_est}")

P_historique = []

P_est = np.eye(n_variable_etat*n_drone)
P_est[0, 0] = sigma_x_init * sigma_x_init #x1
P_est[1, 1] = sigma_x_init * sigma_x_init #y1
P_est[2, 2] = sigma_v_init * sigma_v_init #vx1
P_est[3, 3] = sigma_v_init * sigma_v_init #vy1
P_est[4, 4] = sigma_a_init * sigma_a_init #ax1
P_est[5, 5] = sigma_a_init * sigma_a_init #ay1
P_est[6, 6] = sigma_b_init * sigma_b_init #bx1
P_est[7, 7] = sigma_b_init * sigma_b_init #by1
P_est[8, 8] = sigma_x_init * sigma_x_init #y2
P_est[9, 9] = sigma_x_init * sigma_x_init #y2
P_est[10, 10] = sigma_v_init * sigma_v_init #vx2
P_est[11, 11] = sigma_v_init * sigma_v_init #vy2
P_est[12, 12] = sigma_a_init * sigma_a_init #ax2
P_est[13, 13] = sigma_a_init * sigma_a_init #ay2
P_est[14, 14] = sigma_b_init * sigma_b_init #bx2
P_est[15, 15] = sigma_b_init * sigma_b_init #by2
P_est[16, 16] = sigma_x_init * sigma_x_init #x3
P_est[17, 17] = sigma_x_init * sigma_x_init #y3
P_est[18, 18] = sigma_v_init * sigma_v_init #vx3
P_est[19, 19] = sigma_v_init * sigma_v_init #vy3
P_est[20, 20] = sigma_a_init * sigma_a_init #ax3
P_est[21, 21] = sigma_a_init * sigma_a_init #ay3
P_est[22, 22] = sigma_b_init * sigma_b_init #bx3
P_est[23, 23] = sigma_b_init * sigma_b_init #by3

P_historique.append(P_est)

I_kalman = np.eye(n_variable_etat*n_drone)

Q_kalman = np.eye(n_variable_etat*n_drone)
Q_kalman[0, 0] = sigma_x_1*sigma_x_1 #x1
Q_kalman[1, 1] = sigma_y_1*sigma_y_1 #y1
Q_kalman[2, 2] = sigma_vx_1*sigma_vx_1 #vx1
Q_kalman[3, 3] = sigma_vy_1*sigma_vy_1 #vy1
Q_kalman[4, 4] = sigma_ax_1*sigma_ax_1 #ax1
Q_kalman[5, 5] = sigma_ay_1*sigma_ay_1 #ay1
Q_kalman[6, 6] = sigma_bx_1*sigma_bx_1 #bx1
Q_kalman[7, 7] = sigma_by_1*sigma_by_1 #by1
Q_kalman[8, 8] = sigma_x_2*sigma_x_2 #x2
Q_kalman[9, 9] = sigma_y_2*sigma_y_2 #y2
Q_kalman[10, 10] = sigma_vx_2*sigma_vx_2 #vx2
Q_kalman[11, 11] = sigma_vy_2*sigma_vy_2 #vy2
Q_kalman[12, 12] = sigma_ax_2*sigma_ax_2 #ax2
Q_kalman[13, 13] = sigma_ay_2*sigma_ay_2 #ay2
Q_kalman[14, 14] = sigma_bx_2*sigma_bx_2 #bx2
Q_kalman[15, 15] = sigma_by_2*sigma_by_2 #by2
Q_kalman[16, 16] = sigma_x_3*sigma_x_3 #x3
Q_kalman[17, 17] = sigma_y_3*sigma_y_3 #y3
Q_kalman[18, 18] = sigma_vx_3*sigma_vx_3 #vx3
Q_kalman[19, 19] = sigma_vy_3*sigma_vy_3 #vy3
Q_kalman[20, 20] = sigma_ax_3*sigma_ax_3 #ax3
Q_kalman[21, 21] = sigma_ay_3*sigma_ay_3 #ay3
Q_kalman[22, 22] = sigma_bx_3*sigma_bx_3 #bx3
Q_kalman[23, 23] = sigma_by_3*sigma_by_3 #by3

R_kalman = np.array([
    [sigma_gps_1*sigma_gps_1, 0, 0, 0, 0, 0, 0], #x1
    [0, sigma_gps_1*sigma_gps_1, 0, 0, 0, 0, 0], #y1
    [0, 0, sigma_acc_2*sigma_acc_2, 0, 0, 0, 0], #ax2
    [0, 0, 0, sigma_acc_2*sigma_acc_2, 0, 0, 0], #ay2
    [0, 0, 0, 0, sigma_dist_12*sigma_dist_12, 0, 0], #d12
    [0, 0, 0, 0, 0, sigma_dist_23*sigma_dist_23, 0], #d23
    [0, 0, 0, 0, 0, 0, sigma_dist_13*sigma_dist_13]]) #d13


F_kalman_1 = np.array([
    [1, 0, dt, 0, 0.5*dt*dt, 0, 0, 0],
    [0, 1, 0, dt, 0, 0.5*dt*dt, 0, 0],
    [0, 0, 1, 0, dt, 0, 0, 0],
    [0, 0, 0, 1, 0, dt, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    ])

F_kalman_2 = np.array([
    [1, 0, dt, 0, 0.5*dt*dt, 0, 0, 0],
    [0, 1, 0, dt, 0, 0.5*dt*dt, 0, 0],
    [0, 0, 1, 0, dt, 0, 0, 0],
    [0, 0, 0, 1, 0, dt, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    ])

F_kalman_3 = np.array([
    [1, 0, dt, 0, 0.5*dt*dt, 0, 0, 0],
    [0, 1, 0, dt, 0, 0.5*dt*dt, 0, 0],
    [0, 0, 1, 0, dt, 0, 0, 0],
    [0, 0, 0, 1, 0, dt, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    ])

F_kalman = block_diag(F_kalman_1, F_kalman_2, F_kalman_3)

B_kalman_1 = np.array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],])

B_kalman_2 = np.array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],])

B_kalman_3 = np.array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],])

B_kalman = np.concatenate((B_kalman_1, B_kalman_2, B_kalman_3), axis=0)


t=dt
step = 0
step_capteur = 0

traj_kalman[0] = X_est.copy()
traj_vrai[0] = X_vrai.copy()

while t<t_max :

    if step<((t_max/dt)/3):
        
        u_vrai = [1.0, 0.0 ,1.0, 0.0 ,1.0, 0.0]

        u_vrai_x1_list.append(1.0)
        u_vrai_y1_list.append(0.0)
        u_kalman = u_vrai.copy()

    if step<(2*(t_max/dt)/3) and step>((t_max/dt)/3):

        omega_x = 5 #rad/s
        omega_y = 1

        phi_x += omega_x * dt
        phi_y += omega_y * dt

        u_vrai = [Ax1*np.cos(phi_x), Ay1*np.sin(phi_y), Ax2*np.cos(phi_x), Ay2*np.sin(phi_y), Ax3*np.cos(phi_x), Ay3*np.sin(phi_y)]
        u_vrai_x1_list.append(Ax1*np.sin(phi_x))
        u_vrai_y1_list.append(Ay1*np.sin(phi_y))
        u_kalman = u_vrai.copy()

    if step>(2*(t_max/dt)/3):
        
        u_vrai = [1.0, 0.0 ,1.0, 0.0 ,1.0, 0.0]

        u_vrai_x1_list.append(1.0)
        u_vrai_y1_list.append(0.0)
        u_kalman = u_vrai.copy()

    #Propagation du vrai 
    w_vrai = np.array([np.random.normal(0, sigma_x_1), np.random.normal(0, sigma_y_1), np.random.normal(0, sigma_vx_1), np.random.normal(0, sigma_vy_1), np.random.normal(0, sigma_ax_1), np.random.normal(0, sigma_ay_1), np.random.normal(0, sigma_bx_1), np.random.normal(0, sigma_by_1), np.random.normal(0, sigma_x_2), np.random.normal(0, sigma_y_2), np.random.normal(0, sigma_vx_2), np.random.normal(0, sigma_vy_2), np.random.normal(0, sigma_ax_2), np.random.normal(0, sigma_ay_2), np.random.normal(0, sigma_bx_2), np.random.normal(0, sigma_by_2), np.random.normal(0, sigma_x_3), np.random.normal(0, sigma_y_3), np.random.normal(0, sigma_vx_3), np.random.normal(0, sigma_vy_3), np.random.normal(0, sigma_ax_3), np.random.normal(0, sigma_ay_3), np.random.normal(0, sigma_bx_3), np.random.normal(0, sigma_by_3) ])
    X_vrai = F_vrai @ X_vrai + B_vrai @ u_vrai + w_vrai

    #Propagation de Kalman
    X_pred = F_kalman @ X_est + B_kalman @ u_kalman
    P_pred = F_kalman @ P_est @ F_kalman.T + Q_kalman

    #Correction de Kalman
    if step % int(dt_capteur / dt) ==0 :
        temps_capteur.append(t)

        d12_vrai = np.sqrt((X_vrai[0]-X_vrai[8])**2 + (X_vrai[1]-X_vrai[9])**2)
        d23_vrai = np.sqrt((X_vrai[8]-X_vrai[16])**2 + (X_vrai[9]-X_vrai[17])**2)
        d13_vrai = np.sqrt((X_vrai[0]-X_vrai[16])**2 + (X_vrai[2]-X_vrai[17])**2)
        
        d12_pred = np.sqrt((X_pred[0]-X_pred[8])**2 + (X_pred[1]-X_pred[9])**2)
        d23_pred = np.sqrt((X_pred[8]-X_pred[16])**2 + (X_pred[9]-X_pred[17])**2)
        d13_pred = np.sqrt((X_pred[0]-X_pred[16])**2 + (X_pred[2]-X_pred[17])**2)

        #Propagation capteur
        X_capteur[0] = X_vrai[0] + np.random.normal(0, sigma_gps_1)
        X_capteur[1] = X_vrai[1] + np.random.normal(0, sigma_gps_1)
        X_capteur[2] = X_vrai[12] + X_vrai[14] + np.random.normal(0, sigma_acc_2)
        X_capteur[3] = X_vrai[12] + X_vrai[15] + np.random.normal(0, sigma_acc_2)
        X_capteur[4] = d12_vrai + np.random.normal(0, sigma_dist_12)
        X_capteur[5] = d23_vrai + np.random.normal(0, sigma_dist_23)
        X_capteur[6] = d13_vrai + np.random.normal(0, sigma_dist_13)


        mesure_kalman = X_capteur.copy()

        h40 = (X_pred[0] - X_pred[8]) / d12_pred
        h41 = (X_pred[1] - X_pred[9]) / d12_pred
        h58 = (X_pred[8] - X_pred[16]) / d23_pred
        h59 = (X_pred[9] - X_pred[17]) / d23_pred
        h60 = (X_pred[0] - X_pred[16]) / d13_pred
        h61 = (X_pred[1] - X_pred[17]) / d13_pred

        H_kalman_1 = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [h40, h41, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [h60, h61, 0, 0, 0, 0, 0, 0]])

        H_kalman_2 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [-h40, -h41, 0, 0, 0, 0, 0, 0],
            [h58, h59, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]])

        H_kalman_3 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-h58, -h59, 0, 0, 0, 0, 0, 0],
            [-h60, -h61, 0, 0, 0, 0, 0, 0]])

        H_kalman = np.concatenate((H_kalman_1, H_kalman_2, H_kalman_3), axis=1)

        
        h_X_pred = np.array([X_pred[0], X_pred[1], X_pred[12] + X_pred[14], X_pred[13] + X_pred[15], d12_pred, d23_pred, d13_pred])
        innov = mesure_kalman - h_X_pred
        S = H_kalman @ P_pred @ H_kalman.T + R_kalman
        S_inv = inv(S)
        K = P_pred @ H_kalman.T @ S_inv
        X_est = X_pred + K @ innov
        P_est = (I_kalman - K @ H_kalman) @ P_pred

        mesures_capteur[step_capteur] = np.array([mesure_kalman])

        step_capteur += 1

    else : 
        X_est = X_pred
        P_est = P_pred

    #Enregistrement des trajecoires

    P_historique.append(P_est.copy())
    traj_vrai[step] = X_vrai
    traj_kalman[step] = X_est
    temps.append(t)
    t = t + dt 
    step += 1

################################
#-----------Affichage-----------
################################

temps_np = np.array(temps)[:step]
temps_capteur_np = np.array(temps_capteur)[:step_capteur]
P_hist_np = np.array(P_historique)[:step]

x_vrai_1 = traj_vrai[:step, 0]
y_vrai_1 = traj_vrai[:step, 1]
vx_vrai_1 = traj_vrai[:step, 2]
vy_vrai_1 = traj_vrai[:step, 3]
ax_vrai_1 = traj_vrai[:step, 4]
ay_vrai_1 = traj_vrai[:step, 5]
bx_vrai_1 = traj_vrai[:step, 6]
by_vrai_1 = traj_vrai[:step, 7]

x_vrai_2 = traj_vrai[:step, 8]
y_vrai_2 = traj_vrai[:step, 9]
vx_vrai_2 = traj_vrai[:step, 10]
vy_vrai_2 = traj_vrai[:step, 11]
ax_vrai_2 = traj_vrai[:step, 12]
ay_vrai_2 = traj_vrai[:step, 13]
bx_vrai_2 = traj_vrai[:step, 14]
by_vrai_2 = traj_vrai[:step, 15]

x_vrai_3 = traj_vrai[:step, 16]
y_vrai_3 = traj_vrai[:step, 17]
vx_vrai_3 = traj_vrai[:step, 18]
vy_vrai_3 = traj_vrai[:step, 19]
ax_vrai_3 = traj_vrai[:step, 20]
ay_vrai_3 = traj_vrai[:step, 21]
bx_vrai_3 = traj_vrai[:step, 22]
by_vrai_3 = traj_vrai[:step, 23]

x_capteur_1 = mesures_capteur[:step_capteur, 0]
y_capteur_1 = mesures_capteur[:step_capteur, 1]

ax_capteur_2 = mesures_capteur[:step_capteur, 2]
ay_capteur_2 = mesures_capteur[:step_capteur, 3]

d12_capteur_2 = mesures_capteur[:step_capteur, 4]
d23_capteur_2 = mesures_capteur[:step_capteur, 5]

x_kalman_1 = traj_kalman[:step, 0]
y_kalman_1 = traj_kalman[:step, 1]
vx_kalman_1 = traj_kalman[:step, 2]
vy_kalman_1 = traj_kalman[:step, 3]
ax_kalman_1 = traj_kalman[:step, 4]
ay_kalman_1 = traj_kalman[:step, 5]
bx_kalman_1 = traj_kalman[:step, 6]
by_kalman_1 = traj_kalman[:step, 7]

x_kalman_2 = traj_kalman[:step, 8]
y_kalman_2 = traj_kalman[:step, 9]
vx_kalman_2 = traj_kalman[:step, 10]
vy_kalman_2 = traj_kalman[:step, 11]
ax_kalman_2 = traj_kalman[:step, 12]
ay_kalman_2 = traj_kalman[:step, 13]
bx_kalman_2 = traj_kalman[:step, 14]
by_kalman_2 = traj_kalman[:step, 15]

x_kalman_3 = traj_kalman[:step, 16]
y_kalman_3 = traj_kalman[:step, 17]
vx_kalman_3 = traj_kalman[:step, 18]
vy_kalman_3 = traj_kalman[:step, 19]
ax_kalman_3 = traj_kalman[:step, 20]
ay_kalman_3 = traj_kalman[:step, 21]
bx_kalman_3 = traj_kalman[:step, 22]
by_kalman_3 = traj_kalman[:step, 23]

# derive_theorique_x = np.ones(int(t_max/dt-1)) * derive_x
# derive_theorique_y = np.ones(int(t_max/dt-1)) * derive_y

plot_drones = True
if plot_drones :

    fig, axs = plt.subplots(4, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Analyse EKF - Drone 1", fontsize=14, fontweight='bold')
    axs = axs.flatten()

    sigma = np.sqrt(P_hist_np[:, 0, 0])
    axs[0].plot(temps_np, x_kalman_1 - x_vrai_1, color='green', label='Estimation EKF')
    #axs[0].scatter(temps_capteur_np, x_capteur_1, color="red", marker="x", label="Mesures Drone 1", linewidths=0.5)
    axs[0].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[0].set_title("Position X", fontsize=10)
    axs[0].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 1, 1])
    axs[1].plot(temps_np, y_kalman_1 - y_vrai_1, color='green', label='Estimation EKF')
    #axs[1].scatter(temps_capteur_np, y_capteur_1, color="red", marker="x", label="Mesures Drone 1", linewidths=0.5)
    axs[1].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[1].set_title("Position Y", fontsize=10)
    axs[1].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 2, 2])
    axs[2].plot(temps_np, vx_kalman_1 - vx_vrai_1, color='green', label='Estimation EKF')
    axs[2].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[2].set_title("Vitesse X", fontsize=10)
    axs[2].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 3, 3])
    axs[3].plot(temps_np, vy_kalman_1 - vy_vrai_1, color='green', label='Estimation EKF')
    axs[3].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[3].set_title("Vitesse Y", fontsize=10)
    axs[3].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 4, 4])
    axs[4].plot(temps_np, ax_kalman_1 - ax_vrai_1, color='green', label='Estimation EKF')
    axs[4].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[4].set_title("Acceleration X", fontsize=10)
    axs[4].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 5, 5])
    axs[5].plot(temps_np, ay_kalman_1 - ay_vrai_1, color='green', label='Estimation EKF')
    axs[5].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[5].set_title("Acceleration Y", fontsize=10)
    axs[5].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 6, 6])
    axs[6].plot(temps_np, bx_kalman_1 - bx_vrai_1, color='green', label='Estimation EKF')
    axs[6].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[6].set_title("Biais X", fontsize=10)
    axs[6].grid(True, linestyle=':', alpha=0.7)
    axs[6].set_xlabel("Temps (s)")

    sigma = np.sqrt(P_hist_np[:, 7, 7])
    axs[7].plot(temps_np, bx_kalman_1 - bx_vrai_1, color='green', label='Estimation EKF')
    axs[7].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[7].set_title("Biais Y", fontsize=10)
    axs[7].grid(True, linestyle=':', alpha=0.7)
    axs[7].set_xlabel("Temps (s)")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.95))

    #############################################################################################

    fig, axs = plt.subplots(4, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Analyse EKF - Drone 2", fontsize=14, fontweight='bold')
    axs = axs.flatten()

    sigma = np.sqrt(P_hist_np[:, 8, 8])
    axs[0].plot(temps_np, x_kalman_2 - x_vrai_2, color='green', label='Estimation EKF')
    axs[0].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[0].set_title("Position X", fontsize=10)
    axs[0].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 9, 9])
    axs[1].plot(temps_np, y_kalman_2 - y_vrai_2, color='green', label='Estimation EKF')
    axs[1].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[1].set_title("Position Y", fontsize=10)
    axs[1].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 10, 10])
    axs[2].plot(temps_np, vx_kalman_2 - vx_vrai_2, color='green', label='Estimation EKF')
    axs[2].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[2].set_title("Vitesse X", fontsize=10)
    axs[2].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 11, 11])
    axs[3].plot(temps_np, vy_kalman_2 - vy_vrai_2, color='green', label='Estimation EKF')
    axs[3].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[3].set_title("Vitesse Y", fontsize=10)
    axs[3].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 12, 12])
    axs[4].plot(temps_np, ax_kalman_2 - ax_vrai_2, color='green', label='Estimation EKF')
    axs[4].scatter(temps_capteur_np, ax_capteur_2, color="red", marker="x", label="Mesures Drone 2", linewidths=0.5)
    axs[4].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[4].set_title("Acceleration X", fontsize=10)
    axs[4].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 13, 13])
    axs[5].plot(temps_np, ay_kalman_2 - ay_vrai_2, color='green', label='Estimation EKF')
    axs[5].scatter(temps_capteur_np, ay_capteur_2, color="red", marker="x", label="Mesures Drone 2", linewidths=0.5)
    axs[5].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[5].set_title("Acceleration Y", fontsize=10)
    axs[5].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 14, 14])
    axs[6].plot(temps_np, bx_kalman_2 - bx_vrai_2, color='green', label='Estimation EKF')
    axs[6].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[6].set_title("Biais X", fontsize=10)
    axs[6].grid(True, linestyle=':', alpha=0.7)
    axs[6].set_xlabel("Temps (s)")

    sigma = np.sqrt(P_hist_np[:, 15, 15])
    axs[7].plot(temps_np, bx_kalman_2 - bx_vrai_2, color='green', label='Estimation EKF')
    axs[7].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[7].set_title("Biais Y", fontsize=10)
    axs[7].grid(True, linestyle=':', alpha=0.7)
    axs[7].set_xlabel("Temps (s)")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.95))

    ####################################################################################################################

    fig, axs = plt.subplots(4, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Analyse EKF - Drone 3", fontsize=14, fontweight='bold')
    axs = axs.flatten()

    sigma = np.sqrt(P_hist_np[:, 16, 16])
    axs[0].plot(temps_np, x_kalman_3 - x_vrai_3, color='green', label='Estimation EKF')
    axs[0].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[0].set_title("Position X", fontsize=10)
    axs[0].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 17, 17])
    axs[1].plot(temps_np, y_kalman_3 - y_vrai_3, color='green', label='Estimation EKF')
    axs[1].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[1].set_title("Position Y", fontsize=10)
    axs[1].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 18, 18])
    axs[2].plot(temps_np, vx_kalman_3 - vx_vrai_3, color='green', label='Estimation EKF')
    axs[2].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[2].set_title("Vitesse X", fontsize=10)
    axs[2].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 19, 19])
    axs[3].plot(temps_np, vy_kalman_3 - vy_vrai_3, color='green', label='Estimation EKF')
    axs[3].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[3].set_title("Vitesse Y", fontsize=10)
    axs[3].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 20, 20])
    axs[4].plot(temps_np, ax_kalman_3 - ax_vrai_3, color='green', label='Estimation EKF')
    axs[4].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[4].set_title("Acceleration X", fontsize=10)
    axs[4].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 21, 21])
    axs[5].plot(temps_np, ay_kalman_3 - ay_vrai_3, color='green', label='Estimation EKF')
    axs[5].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[5].set_title("Acceleration Y", fontsize=10)
    axs[5].grid(True, linestyle=':', alpha=0.7)

    sigma = np.sqrt(P_hist_np[:, 22, 22])
    axs[6].plot(temps_np, bx_kalman_3 - bx_vrai_3, color='green', label='Estimation EKF')
    axs[6].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[6].set_title("Biais X", fontsize=10)
    axs[6].grid(True, linestyle=':', alpha=0.7)
    axs[6].set_xlabel("Temps (s)")

    sigma = np.sqrt(P_hist_np[:, 23, 23])
    axs[7].plot(temps_np, bx_kalman_3 - bx_vrai_3, color='green', label='Estimation EKF')
    axs[7].fill_between(temps_np,- 3*sigma, 3*sigma, color='blue', alpha=0.2, label='Couloir $\pm 3\sigma$')
    axs[7].set_title("Biais Y", fontsize=10)
    axs[7].grid(True, linestyle=':', alpha=0.7)
    axs[7].set_xlabel("Temps (s)")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.95))

###################################################################################################################

plot_traj = True
if plot_traj : 
    plt.figure(figsize=(8,6))

    indice_inf = int((t_max/dt)/3)
    indice_sup = int(indice_inf*2)
    plt.scatter(x_capteur_1, y_capteur_1, color="red", marker="x", label="Mesures Drone 1", linewidths=0.5)
    plt.plot(x_vrai_1, y_vrai_1, marker='^', markevery=[indice_inf], label='Drone 1 vrai', color='black')
    plt.plot(x_vrai_2, y_vrai_2, marker='o', markevery=[indice_inf], label='Drone 2 vrai', color='black')
    plt.plot(x_vrai_3, y_vrai_3, marker='s', markevery=[indice_inf], label='Drone 3 vrai', color='black')
    # plt.plot(x_corrige_2, y_corrige_2, label='Drone 2 corrige', color='orange', linestyle='-')
    plt.plot(x_kalman_1, y_kalman_1, marker='^', markevery=[indice_sup], label='Drone 1 corrige par Kalman', color='green', linestyle='-')
    plt.plot(x_kalman_2, y_kalman_2, marker='o', markevery=[indice_sup], label='Drone 2 corrige par Kalman', color='green', linestyle='-')
    plt.plot(x_kalman_3, y_kalman_3, marker='s', markevery=[indice_sup], label='Drone 3 corrige par Kalman', color='green', linestyle='-')

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.title("Trajectoire des 3 drones")
    plt.legend()
    plt.grid(True)

# plt.figure(figsize=(8,6))


# plt.plot(temps_np, u_vrai_x1_list, label='ax1 commande', color='blue')
# plt.plot(temps_np, u_vrai_y1_list, label='ay1 commande', color='black')

# plt.xlabel("t")
# plt.ylabel("Y")

# plt.title("commande en accélération")
# plt.legend()
# plt.grid(True)

print(X_vrai)
plt.show()
