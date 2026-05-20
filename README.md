#Actuellement le code permet de générer le vrai des 3 drones, le capteur du drone 2 et également la correction par milieu des distances vectorielle relatives du drone 2.

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance 
from numpy.linalg import inv

t_max = 200
dt = 0.1
dt_capteur = 1
n_variables_etat = 4

temps = [0]

################################
#------------SIMU---------------
################################

traj = np.zeros((int(t_max/dt), 5, 4))
traj_kalman = np.zeros((int(t_max/dt), 1, 12))

X_vrai_1 = [-1, 0, 0, 0.1]
X_vrai_2 = [0, 0, 0, 0.1]
X_vrai_3 = [1, 0, 0, 0.1]

F_vrai_1 = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]
)

F_vrai_2 = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]
)

F_vrai_3 = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]
)


################################
#-----------Capteur-------------
################################

X_capteur_2 = X_vrai_2
derive_x = 0.00001
derive_y = 0.000
biais_capteur_2 = [derive_x, derive_y]

F_capteur_2 = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]
)


B_capteur_2 = np.array([
    [0, 0],
    [0, 0],
    [dt,0],
    [0, dt]]
)


################################
#-----------Correction----------
################################

X_corrige_2 = X_capteur_2

distante_vraie_x_21 = X_vrai_2[0] - X_vrai_1[0]
distante_vraie_y_21 = X_vrai_2[1] - X_vrai_1[1]
distante_vraie_x_23 = X_vrai_3[0] - X_vrai_2[0] 
distante_vraie_y_23 = X_vrai_3[1] - X_vrai_2[1]

distante_capteur_x_21 = X_capteur_2[0] - X_vrai_1[0]
distante_capteur_y_21 = X_capteur_2[1] - X_vrai_1[1]
distante_capteur_x_32 = X_vrai_3[0] - X_capteur_2[0]
distante_capteur_y_32 = X_vrai_3[1] - X_capteur_2[1]

distance_vraie_21 = distance.euclidean(X_vrai_2[0:2], X_vrai_1[0:2])
distance_vraie_23 = distance.euclidean(X_vrai_3[0:2], X_vrai_2[0:2])

distance_capteur_21 = distance.euclidean(X_capteur_2[0:2], X_vrai_1[0:2])
distance_capteur_23 = distance.euclidean(X_capteur_2[0:2], X_vrai_3[0:2])

distance_relative_2 = [distante_capteur_x_21 - distante_vraie_x_21, distante_capteur_y_21 - distante_vraie_y_21]

F_corrige_2 = np.array([
    [1/2, 0],
    [0, 1/2],
    [0, 0],
    [0, 0]]
)
erreur_corrige = [distance.euclidean(X_corrige_2[0:1], X_vrai_2[0:1])]
erreur_capteur = [distance.euclidean(X_capteur_2[0:1], X_vrai_2[0:1])]
erreur_vraie = [distance.euclidean(X_vrai_2[0:1], X_vrai_2[0:1])]

######Kalman######
X_pred = X_vrai_1 + X_capteur_2 + X_vrai_3
X_est = X_pred

distance_12_init = np.sqrt((X_vrai_1[0]-X_est[4])**2 + (X_vrai_1[1]-X_est[5])**2)
distance_23_init = np.sqrt((X_vrai_3[0]-X_est[4])**2 + (X_vrai_1[1]-X_est[5])**2)
capteur_kalman = [0, 0, 0, 0]

ax = derive_x
ay = derive_y
u_kalman = [ax, ay]

F_kalman = np.array([
    [1, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0], #x1
    [0, 1, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0], #y1
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #vx1
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], #vy1
    [0, 0, 0, 0, 1, 0, dt, 0, 0, 0, 0, 0], #x2
    [0, 0, 0, 0, 0, 1, 0, dt, 0, 0, 0, 0], #y2
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #vx2
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #vy2
    [0, 0, 0, 0, 0, 0, 0, 0, 1, dt, 0, 0], #x3
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, dt, 0], #y3
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], #vx3
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]] #vy3
)

P_est = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
])

B_kalman = np.array([
    [0,0],
    [0,0],
    [dt,0],
    [0,dt],
    [0,0],
    [0,0],
    [dt,0],
    [0,dt],
    [0,0],
    [0,0],
    [dt,0],
    [0,dt],
])

Q_kalman = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
])

R_kalman = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])

I_kalman = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
])
t=dt
step = 1

etat = np.array([X_vrai_1, X_vrai_2, X_vrai_3, X_capteur_2, X_corrige_2])
etat_kalman = np.array([X_pred])

traj_kalman[0] = etat_kalman
traj[0] = etat

while t<(t_max-dt) :
    #### Constrcution du vrai
    X_vrai_1 = F_vrai_1 @ X_vrai_1
    X_vrai_2 = F_vrai_2 @ X_vrai_2
    X_vrai_3 = F_vrai_3 @ X_vrai_3

    #### Constrcution du capteur
    X_capteur_2 = F_capteur_2 @ X_capteur_2 + B_capteur_2 @ biais_capteur_2

    #### Construction de la correction
    distante_vraie_x_21 = X_vrai_2[0] - X_vrai_1[0]
    distante_vraie_y_21 = X_vrai_2[1] - X_vrai_1[1]
    distante_vraie_x_23 = X_vrai_3[0] - X_vrai_2[0] 
    distante_vraie_y_23 = X_vrai_3[1] - X_vrai_2[1]

    distante_capteur_x_21 = X_capteur_2[0] - X_vrai_1[0]
    distante_capteur_y_21 = X_capteur_2[1] - X_vrai_1[1]
    distante_capteur_x_32 = X_vrai_3[0] - X_capteur_2[0]
    distante_capteur_y_32 = X_vrai_3[1] - X_capteur_2[1]

    distance_relative_2 = [distante_capteur_x_21 - distante_vraie_x_21, distante_capteur_y_21 - distante_vraie_y_21]

    X_corrige_2 = X_capteur_2 - F_corrige_2 @ distance_relative_2

    #### Kalman
    
    #Prédiction
    X_pred = F_kalman @ X_est + B_kalman @ u_kalman
    P_pred = F_kalman @ P_est @ F_kalman.T + Q_kalman

    #Mise à jour
    if t%dt_capteur==0:
        
        capteur_kalman[0:2] = X_capteur_2[0:2]
        capteur_kalman[2] = np.sqrt((X_vrai_1[0]-X_est[4])**2 + (X_vrai_1[1]-X_est[5])**2)
        capteur_kalman[3] = np.sqrt((X_vrai_3[0]-X_est[4])**2 + (X_vrai_3[1]-X_est[5])**2)

        d_hat_12 = np.sqrt((X_est[0]-X_est[4])**2 + (X_est[1]-X_est[5])**2)
        d_hat_23 = np.sqrt((X_est[4]-X_est[8])**2 + (X_est[5]-X_est[9])**2)

        h20 = (X_est[0]-X_est[4])/d_hat_12
        h21 = (X_est[1]-X_est[5])/d_hat_12
        h24 = (X_est[4]-X_est[0])/d_hat_12
        h25 = (X_est[5]-X_est[1])/d_hat_12

        h34 = (X_est[4]-X_est[8])/d_hat_23
        h35 = (X_est[5]-X_est[9])/d_hat_23
        h38 = (X_est[8]-X_est[4])/d_hat_23
        h39 = (X_est[9]-X_est[5])/d_hat_23

        H_kalman = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [h20, h21, 0, 0, h24, h25, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, h34, h35, 0, 0, h38, h39, 0, 0],
        ])

        innov = capteur_kalman - H_kalman @ X_pred
        S = H_kalman @ P_pred @ H_kalman.T + R_kalman
        S_inv = inv(S)
        K = P_pred @ H_kalman.T @ S_inv
        X_est = X_pred + K @ innov
        P_est = (I_kalman - K @ H_kalman) @ P_pred

    else : 
        X_est = X_pred
    

    etat = np.array([X_vrai_1, X_vrai_2, X_vrai_3, X_capteur_2, X_corrige_2])
    etat_kalman = np.array([X_est])

    erreur_corrige.append(distance.euclidean(X_corrige_2[0:2], X_vrai_2[0:2]))
    erreur_capteur.append(distance.euclidean(X_capteur_2[0:2], X_vrai_2[0:2]))
    erreur_vraie.append(distance.euclidean(X_vrai_2[0:2], X_vrai_2[0:2]))
    
    traj_kalman[step] = etat_kalman
    traj[step] = etat
    temps.append(t)
    t = t + dt 
    txt = f"{t:.1f}"
    t = float(txt)
    step+=1

print(f"Taille de traj avant : {traj.shape}")
traj = traj[:-1, :, :]
print(f"Taille de traj après : {traj.shape}")

print(f"Taille de traj_kalman avant : {traj_kalman.shape}")
traj_kalman = traj_kalman[:-1, :, :]
print(f"Taille de traj_kalman après : {traj_kalman.shape}")

x_vrai_1 = traj[:, 0, 0]
y_vrai_1 = traj[:, 0, 1]

x_vrai_2 = traj[:, 1, 0]
y_vrai_2 = traj[:, 1, 1]

x_vrai_3 = traj[:, 2, 0]
y_vrai_3 = traj[:, 2, 1]

x_capteur_2 = traj[:, 3, 0]
y_capteur_2 = traj[:, 3, 1]

x_corrige_2 = traj[:, 4, 0]
y_corrige_2 = traj[:, 4, 1]

x_kalman = traj_kalman[:, 0, 4]
y_kalman = traj_kalman[:, 0, 5]


plt.figure(figsize=(8,6))

plt.plot(x_vrai_1, y_vrai_1, label='Drone 1 vrai', color='black')
plt.plot(x_vrai_2, y_vrai_2, label='Drone 2 vrai', color='black')
plt.plot(x_capteur_2, y_capteur_2, label='Drone 2 capteur', color='blue')
plt.plot(x_corrige_2, y_corrige_2, label='Drone 2 corrige', color='orange', linestyle='-')
plt.plot(x_kalman, y_kalman, label='Drone 2 corrige par Kalman', color='green', linestyle='-')
plt.plot(x_vrai_3, y_vrai_3, label='Drone 3 vrai', color='black')

plt.xlabel("X")
plt.ylabel("Y")

plt.title("Trajectoire des 3 drones")
plt.legend()
plt.grid(True)

plt.figure(figsize=(8,6))

plt.plot(temps, erreur_corrige, label='Erreur Corrigé', color='green')
plt.plot(temps, erreur_capteur, label='Erreur capteur', color='blue')
plt.plot(temps, erreur_vraie, label='Erreur vrai', color='black')
plt.title("Erreur de distance par rapport au vrai entre le corrigé et le capteur Drone 2")

plt.xlabel("Temps (s)")
plt.ylabel("Erreur en (m)")

plt.grid(True)
plt.legend()

plt.show()






