-----MODE TRAINING-----
Préparation des données...
Initialisation du réseau...
Traceback (most recent call last):
  File "/home/gsionsua/Work_bis/KalmanNet/main.py", line 20, in <module>
    main()
  File "/home/gsionsua/Work_bis/KalmanNet/main.py", line 12, in main
    run_training(cfg)
  File "/home/gsionsua/Work_bis/KalmanNet/train.py", line 16, in run_training
    model = KalmanNet(cfg).to(cfg.device)
  File "/home/gsionsua/Work_bis/KalmanNet/src/models/kalmannet.py", line 9, in __init__
    super().__init__()
TypeError: BaseFilter.__init__() missing 3 required positional arguments: 'state_dim', 'obs_dim', and 'device'
(Environnement) scvmpr10.fr.mbda.priv:/home/gsionsua/Work_bis/KalmanNet $
