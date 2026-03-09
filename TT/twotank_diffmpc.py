#cleaned up from JK_diffmpc_test.py
import torch
from torch.autograd import Function, Variable, grad
from torch.nn.parameter import Parameter

import numpy as np


import numdifftools as nd

import gc
import os

from mpc import mpc, util, pnqp
from mpc.dynamics import NNDynamics, AffineDynamics
from mpc.lqr_step import LQRStep
from mpc.mpc import GradMethods, QuadCost, LinDx
import matplotlib.pyplot as plt

p = 1e-5 # add for numerical issues with sqrt
def ode_equations(x, u, c, xmin=torch.Tensor([0.0]), xmax=torch.Tensor([1.0]), umin=torch.Tensor([0.0]), umax=torch.Tensor([1.0])):

    c1 = c[:, [0]]
    c2 = c[:, [1]]

    # heights in tanks
    h1 = torch.clip(x[:, [0]], min=0, max=1.0)
    h2 = torch.clip(x[:, [1]], min=0, max=1.0)
    # Inputs (2): pump and valve
    pump = torch.clip(u[:, [0]], min=0, max=1.0)
    valve = torch.clip(u[:, [1]], min=0, max=1.0)
    # equations
    dhdt1 = c1 * (1.0 - valve) * pump - c2 * torch.sqrt(h1+p)
    dhdt2 = c1 * valve * pump + c2 * torch.sqrt(h1+p) - c2 * torch.sqrt(h2+p)
    return torch.cat([dhdt1, dhdt2], dim=-1)


class TwoTankDx(torch.nn.Module):
    def __init__(self, c=None, dt=0.5):
        super().__init__()

        self.n_state = 2
        self.n_ctrl = 2
        self.c = c  # design variables
        self.dt = dt # TODO: get the right value

    def forward(self,x,u):

        delta_x = self.dt*ode_equations(x, u, self.c)
        return x + delta_x


class DiffTwoTankSolver(torch.nn.Module):
    def __init__(self, T, dt, u_lower=0.0, u_upper=1.0, lqr_iter = 100, verbose = 1):
        super().__init__()

        self.n_state = 2
        self.n_ctrl = 2
        self.dt = dt # TODO: get the right value
        self.T = T
        self.lqr_iter = lqr_iter

        self.u_lower = u_lower
        self.u_upper = u_upper

        Ix = torch.eye(self.n_state)
        Iu = torch.eye(self.n_ctrl)
        self.Q = torch.block_diag(Ix,0.001*Iu)   # Increasing 0.001 to 0.01 causes termination in <20 iters and no non-convergence warning

        self.diff_solver = mpc.MPC(
            self.n_state, self.n_ctrl, self.T, self.u_lower, self.u_upper,
            lqr_iter=self.lqr_iter,
            verbose=verbose,
            exit_unconverged=False,
            backprop=True,
            max_linesearch_iter=1,
            grad_method = GradMethods.AUTO_DIFF
        )

    # Inputs match those of prev learned control policy in order:
    # Design params c    (batch thereof - 2d)
    # Initial state x0   (batch thereof - 2d)
    # Reference traj r   (batch thereof - 3d)
    def forward(self,c,x0,r):
        n_batch = r.shape[0]
        dx = TwoTankDx(c, self.dt)

        r = torch.cat((r,0*r),dim=2)    # a single state+ctrl traj
        rbat_shaped = r.permute(1,0,2)
        qbat  = -rbat_shaped   # TODO: check if should be -2

        Qbat = self.Q.repeat(n_batch,1,1).repeat(self.T,1,1,1)
        cost = QuadCost(Qbat.double(), qbat.double())   #line 135 test_mpc.py

        x0 = x0.double()

        x_lqr, u_lqr, objs_lqr = self.diff_solver(x0, cost, dx)
        x = x_lqr.permute(1,0,2).float()
        u = u_lqr.permute(1,0,2).float()

        return x,u




if __name__=='__main__':


    T = 20
    dt = 0.5
    n_state = 2
    n_ctrl  = 2

    diff_tt_solver = DiffTwoTankSolver(T, dt, lqr_iter = 100)



    ########## TEST on the expected distributions of c and xr for training ########

    n_test = 32

    x0 =  torch.zeros(n_test, 2).double()

    x2min = 0.2
    x1minrat = 0.5
    x2 = (1.0-x2min)*torch.rand(n_test,1)+x2min;   x1 = x2*(  (1.0-x1minrat)*torch.rand(n_test,1) + x1minrat  )
    xf = torch.cat((x1,x2),dim=1)
    xr = xf.unsqueeze(1).repeat(1,T,1)

    #cmax = 0.10
    #cmin = 0.01
    #c = (cmax-cmin)*torch.rand(n_test,2) + cmin
    c = torch.Tensor([0.32,0.16])*torch.ones(n_test,n_state)
    #c = 4.0*torch.Tensor([0.08,0.04])*torch.ones(n_test,n_state)

    print("x0.shape")
    print( x0.shape )

    print("xr.shape")
    print( xr.shape )

    print("c.shape")
    print( c.shape )


    x, u = diff_tt_solver(c,x0,xr)


    print("\n\n\n\n\n\n\n\n\n\n\n\n")
    for i in range(len(c)):

        #x, u = diff_tt_solver(c[i].unsqueeze(0),x0[i].unsqueeze(0),xr[i].unsqueeze(0))

        plt.plot( [xf[i][0]], [xf[i][1]],  'r*')#, label=r"Target point" )
        plt.plot(    x[i][:,0].detach(),   x[i][:,1].detach(),  'b*-')#, label=r"Predicted trajectory, c = [{:.4f},  {:.4f}]".format(c[i][0], c[i][1]) )

    plt.xlim(0,1.0)
    plt.ylim(0,1.0)
    plt.legend()
    plt.show()
