"""# **Class: Extended Kalman Filter**
Theoretical Non Linear Kalman
"""
import torch

from Simulations.Lorenz_Atractor.parameters import getJacobian

class ExtendedKalmanFilter:

    def __init__(self, SystemModel, args):
        # Device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # process model
        self.f = SystemModel.f
        self.m = SystemModel.m
        self.Q = SystemModel.Q.to(self.device)
        # observation model
        self.h = SystemModel.h
        self.n = SystemModel.n
        self.R = SystemModel.R.to(self.device)
        # sequence length (use maximum length if random length case)
        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

        # ---- Command interface (off by default) ----
        # When enabled via SetCommand(...), the predict step uses
        # f_t(x) = F x + B u(t), rebuilt each time step. The state Jacobian is F.
        self.commanded = False
        # Optional analytic observation Jacobian h(x, jacobian=True) -> (y, H).
        # If provided, it is used instead of the numerical getJacobian for h.
        self.h_jac = getattr(SystemModel, 'h_jac', None)
  
    # Predict
    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.m1x_posterior).to(self.device)

        # ---- Jacobians ----
        if getattr(self, 'commanded', False):
            # f_t(x) = F x + B u(t)  ->  d f / d x = F (exact, constant).
            batch = self.m1x_posterior.shape[0]
            F_batched = self.F_matrix.to(self.device).expand(batch, self.m, self.m)
            if self.h_jac is not None:
                # analytic observation Jacobian: h(x, jacobian=True) -> (y, H)
                _, H_batched = self.h_jac(self.m1x_prior, jacobian=True)
                H_batched = H_batched.to(self.device)
            else:
                H_batched = getJacobian(self.m1x_prior, self.h)
            self.UpdateJacobians(F_batched, H_batched)
        else:
            # Original repo behaviour: numerical Jacobians of f and h.
            self.UpdateJacobians(getJacobian(self.m1x_posterior, self.f),
                                 getJacobian(self.m1x_prior, self.h))

        # Predict the 2-nd moment of x
        self.m2x_prior = torch.bmm(self.batched_F, self.m2x_posterior)
        self.m2x_prior = torch.bmm(self.m2x_prior, self.batched_F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)
        # Predict the 2-nd moment of y
        self.m2y = torch.bmm(self.batched_H, self.m2x_prior)
        self.m2y = torch.bmm(self.m2y, self.batched_H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.bmm(self.m2x_prior, self.batched_H_T)
        self.KG = torch.bmm(self.KG, torch.inverse(self.m2y))

        #Save KalmanGain
        self.KG_array[:,:,:,self.i] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.bmm(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.bmm(self.m2y, torch.transpose(self.KG, 1, 2))
        self.m2x_posterior = self.m2x_prior - torch.bmm(self.KG, self.m2x_posterior)

    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior

    #########################

    def UpdateJacobians(self, F, H):
        self.batched_F = F.to(self.device)
        self.batched_F_T = torch.transpose(F,1,2)
        self.batched_H = H.to(self.device)
        self.batched_H_T = torch.transpose(H,1,2)
    
    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):

            self.m1x_0_batch = m1x_0_batch # [batch_size, m, 1]
            self.m2x_0_batch = m2x_0_batch # [batch_size, m, m]

    #########################
    ### Command interface ###
    #########################
    def SetCommand(self, F_matrix, B_matrix, U, make_f_commanded):
        """
        Enable commanded prediction f_t(x) = F x + B u(t).
          F_matrix, B_matrix : constant dynamics matrices.
          U                  : command sequence [dim_u, T] (MUST be the SAME U
                               used to generate the data, for consistency).
          make_f_commanded   : factory; make_f_commanded(u_t) -> f.
        """
        self.F_matrix = F_matrix
        self.B_matrix = B_matrix
        self.U = U
        self.make_f_commanded = make_f_commanded
        self.commanded = True

    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, y):
        """
        input y: batch of observations [batch_size, n, T]
        """
        y = y.to(self.device)
        self.batch_size = y.shape[0] # batch size
        T = y.shape[2] # sequence length (maximum length if randomLength=True)

        # Pre allocate KG array
        self.KG_array = torch.zeros([self.batch_size,self.m,self.n,T]).to(self.device)
        self.i = 0 # Index for KG_array alocation

        # Allocate Array for 1st and 2nd order moments (use zero padding)
        self.x = torch.zeros(self.batch_size, self.m, T).to(self.device)
        self.sigma = torch.zeros(self.batch_size, self.m, self.m, T).to(self.device)
            
        # Set 1st and 2nd order moments for t=0
        self.m1x_posterior = self.m1x_0_batch.to(self.device)
        self.m2x_posterior = self.m2x_0_batch.to(self.device)

        # Generate in a batched manner
        for t in range(0, T):
            # If commanded, rebuild f for this step: f_t(x) = F x + B u(t).
            if getattr(self, 'commanded', False):
                self.f = self.make_f_commanded(self.U[:, t])
            yt = torch.unsqueeze(y[:, :, t],2)
            xt,sigmat = self.Update(yt)
            self.x[:, :, t] = torch.squeeze(xt,2)
            self.sigma[:, :, :, t] = sigmat