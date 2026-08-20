== Chargement des modeles ==
   A (etroit)       : ['archi1', 'archi2']
   B (randomise)    : ['archi1', 'archi2']

== Evaluation par famille (20 trajectoires each) ==

  3 phases (reference in-distrib.)
     EKF   MSE pos d2 = 6.5175
     A (etroit)     archi1 : MSE=33.0109  Delta= +7.05 dB  div=0%  <-- battu par l'EKF
     A (etroit)     archi2 : MSE=18.1277  Delta= +4.44 dB  div=0%  <-- battu par l'EKF
     B (randomise)  archi1 : MSE=23.8714  Delta= +5.64 dB  div=5%  <-- battu par l'EKF
     B (randomise)  archi2 : MSE=18.7822  Delta= +4.60 dB  div=0%  <-- battu par l'EKF

  Creneaux (bang-bang)
     EKF   MSE pos d2 = 13.1040
     A (etroit)     archi1 : MSE=17033.0679  Delta=+31.14 dB  div=45%  <-- battu par l'EKF
     A (etroit)     archi2 : MSE=168.5076  Delta=+11.09 dB  div=25%  <-- battu par l'EKF
     B (randomise)  archi1 : MSE=50.7871  Delta= +5.88 dB  div=5%  <-- battu par l'EKF
     B (randomise)  archi2 : MSE=14.2718  Delta= +0.37 dB  div=0%  <-- battu par l'EKF

  Chirp
     EKF   MSE pos d2 = 8.4773
     A (etroit)     archi1 : MSE=101.4985  Delta=+10.78 dB  div=5%  <-- battu par l'EKF
     A (etroit)     archi2 : MSE=22.7334  Delta= +4.28 dB  div=0%  <-- battu par l'EKF
     B (randomise)  archi1 : MSE=22.4289  Delta= +4.23 dB  div=0%  <-- battu par l'EKF
     B (randomise)  archi2 : MSE=17.0884  Delta= +3.04 dB  div=0%  <-- battu par l'EKF

  Virage coordonne
     EKF   MSE pos d2 = 8.7585
     A (etroit)     archi1 : MSE=7509.8719  Delta=+29.33 dB  div=50%  <-- battu par l'EKF
     A (etroit)     archi2 : MSE=59.1653  Delta= +8.30 dB  div=15%  <-- battu par l'EKF
     B (randomise)  archi1 : MSE=28.9977  Delta= +5.20 dB  div=0%  <-- battu par l'EKF
     B (randomise)  archi2 : MSE=14.8483  Delta= +2.29 dB  div=0%  <-- battu par l'EKF

  Stop-and-go
     EKF   MSE pos d2 = 11.3332
     A (etroit)     archi1 : MSE=7626.2636  Delta=+28.28 dB  div=40%  <-- battu par l'EKF
     A (etroit)     archi2 : MSE=75.6061  Delta= +8.24 dB  div=0%  <-- battu par l'EKF
     B (randomise)  archi1 : MSE=19.5056  Delta= +2.36 dB  div=0%  <-- battu par l'EKF
     B (randomise)  archi2 : MSE=11.3898  Delta= +0.02 dB  div=0%  <-- battu par l'EKF

== Balayage en amplitude (Creneaux (bang-bang)) ==
  A=0.5 | EKF=   8.003 | A-1=+26.07dB | A-2= +3.59dB | B-1= +5.34dB | B-2= -0.94dB
  A=1.0 | EKF=  13.850 | A-1=+26.23dB | A-2= +8.65dB | B-1= +4.79dB | B-2= -3.09dB
  A=1.5 | EKF=   8.800 | A-1=+30.60dB | A-2=+14.17dB | B-1= +5.81dB | B-2= -0.71dB
  A=2.0 | EKF=   9.068 | A-1=+33.73dB | A-2=+17.16dB | B-1= +5.93dB | B-2= -2.34dB
  A=2.5 | EKF=   9.175 | A-1=+35.24dB | A-2=+19.25dB | B-1= +8.22dB | B-2= -2.63dB
  A=3.0 | EKF=  19.063 | A-1=+33.65dB | A-2=+17.53dB | B-1= +5.57dB | B-2= -4.59dB
