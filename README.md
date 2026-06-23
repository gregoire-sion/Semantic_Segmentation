"""
main_drones3.py  -- CENTRALIZED 3-drone collaborative navigation with KalmanNet.

Adapted from main_lor_DT_NLobs.py (the non-linear-observation case),
because the inter-drone ranges make h() non-linear.

Run:  python3 main_drones3.py
"""
import torch
from datetime import datetime

from Filters.EKF_test import EKFTest
from Simulations.Extended_sysmdl import SystemModel
from Simulations.utils import DataGen
import Simulations.config as config

# >>> CHANGED: import the 3-drone model instead of the Lorenz one
from Simulations.Drones3.parameters import (
    m1x_0, m2x_0, m, n, f, h, getJacobian, Q_structure, R_structure
)

from Pipelines.Pipeline_EKF import Pipeline_EKF
from KNet.KalmanNet_nn import KalmanNetNN

print("Pipeline Start")
today = datetime.today(); now = datetime.now()
strTime = today.strftime("%m.%d.%y") + "_" + now.strftime("%H:%M:%S")
print("Current Time =", strTime)

###################
###  Settings   ###
###################
args = config.general_settings()
### dataset sizes (slide-8 numbers; scale down for a first run)
args.N_E = 1000      # training sequences
args.N_CV = 100      # validation
args.N_T = 200       # test
args.T = 50          # sequence length (train/cv)
args.T_test = 50

### KalmanNet capacity
args.in_mult_KNet = 40
args.out_mult_KNet = 5

### training
args.use_cuda = False     # set True if you have a GPU
args.n_steps = 2000
args.n_batch = 50
args.lr = 1e-4
args.wd = 1e-4
args.CompositionLoss = True
args.alpha = 0.5

device = torch.device('cuda' if (args.use_cuda and torch.cuda.is_available()) else 'cpu')
print("Using", "GPU" if device.type == 'cuda' else "CPU")

path_results = 'KNet/'
DatafolderName = 'Simulations/Drones3/data/'
dataFileName = 'data_drones3_T50.pt'

#############################
### Noise levels          ###
#############################
# Global scaling: q2 is process-noise scale, r2 observation-noise scale.
# The PER-CHANNEL structure (incl. the big D2-accel rows) lives in
# R_structure / Q_structure inside parameters.py.
r2 = torch.tensor([1.0])
q2 = torch.tensor([1.0])

Q_true = q2[0] * Q_structure
R_true = r2[0] * R_structure

#############################
### Build system models   ###
#############################
# Ground-truth model that GENERATES the data (well specified)
sys_model = SystemModel(f, Q_true, h, R_true, args.T, args.T_test, m, n)
sys_model.InitSequence(m1x_0, m2x_0)

print("Start Data Gen")
DataGen(args, sys_model, DatafolderName + dataFileName)
print("Data Load:", dataFileName)
[train_input_long, train_target_long, cv_input, cv_target,
 test_input, test_target, _, _, _] = torch.load(DatafolderName + dataFileName, map_location=device)

train_target = train_target_long[:, :, 0:args.T]
train_input  = train_input_long[:, :, 0:args.T]

print("trainset:", train_target.size())
print("cvset:", cv_target.size())
print("testset:", test_target.size())

########################
### Evaluate EKF     ###
########################
# IMPORTANT first step: run the EKF on WELL-SPECIFIED data to confirm the
# joint state is observable. If the EKF diverges here, D2/D3 are under-
# determined (no absolute position) -- fix geometry/anchoring BEFORE KNet.
print("Evaluate EKF (observability sanity check)")
[MSE_EKF_lin_arr, MSE_EKF_lin_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = \
    EKFTest(args, sys_model, test_input, test_target)

##########################
### Evaluate KalmanNet ###
##########################
print("KalmanNet start")
KalmanNet_model = KalmanNetNN()
KalmanNet_model.NNBuild(sys_model, args)
print("Trainable params:",
      sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))

KalmanNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
KalmanNet_Pipeline.setssModel(sys_model)
KalmanNet_Pipeline.setModel(KalmanNet_model)
KalmanNet_Pipeline.setTrainingParams(args)

print("Composition Loss:", args.CompositionLoss)
[MSE_cv_lin_epoch, MSE_cv_dB_epoch, MSE_train_lin_epoch, MSE_train_dB_epoch] = \
    KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target,
                               train_input, train_target, path_results)

[MSE_test_lin_arr, MSE_test_lin_avg, MSE_test_dB_avg, knet_out, RunTime] = \
    KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)

print("Done. EKF MSE [dB]:", float(MSE_EKF_dB_avg),
      " | KNet MSE [dB]:", float(MSE_test_dB_avg))
