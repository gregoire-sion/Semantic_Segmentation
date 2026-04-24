(Environnement) scvmpr10.fr.mbda.priv:/home/gsionsua/Work_bis/KalmanNet $python main.py 
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
  File "/home/gsionsua/Work_bis/KalmanNet/src/models/kalmannet.py", line 21, in __init__
    self.rnn = nn.GRU(input_size=self.obs_dim, hidden_size=config.hidden_dim, num_layer=1, batch_first=True)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 900, in __init__
    super(GRU, self).__init__('GRU', *args, **kwargs)
TypeError: RNNBase.__init__() got an unexpected keyword argument 'num_layer'
