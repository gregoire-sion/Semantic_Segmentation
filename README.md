== Run : baseline_narrow_archi2_seed42 ==
   sortie  : ./runs/baseline_narrow_archi2_seed42
   device  : cuda
   config  : noise_sweep=False cmd_randomize=False init_offset=0.3
   graines : train=42 val=43 test=141

== Génération des données ==
>> Dataset sauvegardé -> ./runs/baseline_narrow_archi2_seed42/dataset.npz
   train=400  val=80  test=50
   62.8 s

== Entraînement archi2 ==
   paramètres : 24107793
[archi2] epoch   0 | train 4.5580e+07 | val 2.4723e-01 | best 2.4723e-01
[archi2] epoch   5 | train 7.1317e-01 | val 9.1361e-02 | best 8.3679e-02
[archi2] epoch  10 | train 6.7440e-01 | val 8.5288e-02 | best 8.3679e-02
[archi2] epoch  15 | train 6.5094e-01 | val 8.8591e-02 | best 8.2391e-02
[archi2] epoch  20 | train 6.2919e-01 | val 8.1348e-02 | best 7.9602e-02
[archi2] epoch  25 | train 6.5406e-01 | val 7.7902e-02 | best 7.7902e-02
[archi2] epoch  30 | train 5.7720e-01 | val 8.9756e-02 | best 7.7902e-02
[archi2] epoch  35 | train 5.7812e-01 | val 8.4868e-02 | best 7.7902e-02
[archi2] epoch  39 | train 5.6161e-01 | val 8.1408e-02 | best 7.7902e-02
[archi2] modèle sauvé -> ./runs/baseline_narrow_archi2_seed42/knet_archi2.pt
   loss train divisée par 8 (fenêtres TBPTT) pour l'affichage

== Évaluation en distribution ==
groupe             MSE KNet      MSE EKF    gain dB
position             0.9762       3.1387      -5.07
vitesse              0.0275       0.0851      -4.90
acceleration         0.0040       0.0140      -5.49
biais                0.0598       0.0651      -0.38

   Delta_dB (position) : -5.41 +/- 1.81 dB (IC95, n=50)
   Delta_dB < 0  =>  KalmanNet meilleur que l'EKF

== Manifeste  -> ./runs/baseline_narrow_archi2_seed42/manifest.json
== Figure     -> ./runs/baseline_narrow_archi2_seed42/loss_archi2.png
== Checkpoint -> ./runs/baseline_narrow_archi2_seed42/knet_archi2.pt


== Run : baseline_narrow_archi2_seed1234 ==
   sortie  : ./runs/baseline_narrow_archi2_seed1234
   device  : cuda
   config  : noise_sweep=False cmd_randomize=False init_offset=0.3
   graines : train=1234 val=1235 test=1333

== Génération des données ==
>> Dataset sauvegardé -> ./runs/baseline_narrow_archi2_seed1234/dataset.npz
   train=400  val=80  test=50
   112.7 s

== Entraînement archi2 ==
   paramètres : 24107793
[archi2] epoch   0 | train 1.6840e+11 | val 1.5878e-01 | best 1.5878e-01
[archi2] epoch   5 | train 7.3711e-01 | val 8.5678e-02 | best 8.5678e-02
[archi2] epoch  10 | train 6.9540e-01 | val 8.8265e-02 | best 8.5678e-02
[archi2] epoch  15 | train 6.6412e-01 | val 1.0087e-01 | best 8.5678e-02
[archi2] epoch  20 | train 6.2068e-01 | val 7.9550e-02 | best 7.9550e-02
[archi2] epoch  25 | train 6.1173e-01 | val 8.4831e-02 | best 7.9550e-02
[archi2] epoch  30 | train 5.8578e-01 | val 8.5414e-02 | best 7.9550e-02
[archi2] epoch  35 | train 5.8621e-01 | val 8.5340e-02 | best 7.9550e-02
[archi2] epoch  39 | train 5.6991e-01 | val 9.3378e-02 | best 7.9550e-02
[archi2] modèle sauvé -> ./runs/baseline_narrow_archi2_seed1234/knet_archi2.pt
   loss train divisée par 8 (fenêtres TBPTT) pour l'affichage

== Évaluation en distribution ==
groupe             MSE KNet      MSE EKF    gain dB
position             0.8801       3.2355      -5.65
vitesse              0.0290       0.0888      -4.85
acceleration         0.0039       0.0145      -5.71
biais                0.0744       0.0771      -0.16

   Delta_dB (position) : -5.51 +/- 1.70 dB (IC95, n=50)
   Delta_dB < 0  =>  KalmanNet meilleur que l'EKF

== Manifeste  -> ./runs/baseline_narrow_archi2_seed1234/manifest.json
== Figure     -> ./runs/baseline_narrow_archi2_seed1234/loss_archi2.png
== Checkpoint -> ./runs/baseline_narrow_archi2_seed1234/knet_archi2.pt

== Run : baseline_narrow_archi2_seed7 ==
   sortie  : ./runs/baseline_narrow_archi2_seed7
   device  : cuda
   config  : noise_sweep=False cmd_randomize=False init_offset=0.3
   graines : train=7 val=8 test=106

== Génération des données ==
>> Dataset sauvegardé -> ./runs/baseline_narrow_archi2_seed7/dataset.npz
   train=400  val=80  test=50
   297.2 s

== Entraînement archi2 ==
   paramètres : 24107793
[archi2] epoch   0 | train 2.2323e+10 | val 1.1605e-01 | best 1.1605e-01
[archi2] epoch   5 | train 7.4734e-01 | val 9.4306e-02 | best 9.4306e-02
[archi2] epoch  10 | train 6.9263e-01 | val 9.4989e-02 | best 9.4306e-02
[archi2] epoch  15 | train 6.5825e-01 | val 9.2126e-02 | best 9.2126e-02
[archi2] epoch  20 | train 6.2543e-01 | val 9.3631e-02 | best 9.0420e-02
[archi2] epoch  25 | train 6.0700e-01 | val 1.0181e-01 | best 9.0211e-02
[archi2] epoch  30 | train 5.9424e-01 | val 9.8830e-02 | best 8.9875e-02
[archi2] epoch  35 | train 5.6943e-01 | val 9.6381e-02 | best 8.9875e-02
[archi2] epoch  39 | train 5.7478e-01 | val 9.4576e-02 | best 8.7451e-02
[archi2] modèle sauvé -> ./runs/baseline_narrow_archi2_seed7/knet_archi2.pt
   loss train divisée par 8 (fenêtres TBPTT) pour l'affichage

== Évaluation en distribution ==
groupe             MSE KNet      MSE EKF    gain dB
position             0.7432       3.2529      -6.41
vitesse              0.0215       0.0874      -6.08
acceleration         0.0038       0.0140      -5.65
biais                0.0578       0.0600      -0.16

   Delta_dB (position) : -6.39 +/- 1.66 dB (IC95, n=50)
   Delta_dB < 0  =>  KalmanNet meilleur que l'EKF

== Manifeste  -> ./runs/baseline_narrow_archi2_seed7/manifest.json
== Figure     -> ./runs/baseline_narrow_archi2_seed7/loss_archi2.png
== Checkpoint -> ./runs/baseline_narrow_archi2_seed7/knet_archi2.pt



