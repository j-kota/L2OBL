import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from twotank_diffmpc import DiffTwoTankSolver
import torch
import matplotlib.pyplot as plt

class PSOmodel():
    def __init__(self, cmin, cmax, T=20, dt=0.5, rho = 1e3, n_particles = 12, pso_iter = 20, lqr_iter = 100):
        super().__init__()
        self.T = T
        self.pso_iter = pso_iter
        self.lqr_iter = lqr_iter
        self.n_particles = n_particles
        self.rho    = rho

        self.diff_tt_solver = DiffTwoTankSolver(T, dt, lqr_iter = self.lqr_iter, verbose=0)

        bounds = (cmin*np.ones(2), cmax*np.ones(2))
        options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9}
        self.pyswarms_solver = GlobalBestPSO(n_particles=self.n_particles, dimensions=2, bounds=bounds, options=options)


        """
        Calling solve() will populate these lists with the histories of that latest solver call
        """
        self.cost_history = None
        self.mean_pbest_history = None
        self.mean_neighbor_history = None
        self.pos_history = None
        self.velocity_history = None


    def design_objective(self, c, xf):

        c  = torch.Tensor(c)
        xf = torch.Tensor(xf)
        x0 = torch.zeros(c.shape)

        xf = xf.unsqueeze(0).repeat(len(c),1)
        xr = xf.unsqueeze(1).repeat(1,self.T,1)

        x, u = self.diff_tt_solver(c,x0,xr)

        f = c.sum(1) + self.rho*( (x[:,-1,:] - xf)**2 ).sum(1)
        return f.detach().numpy()


    def design_objective_split(self, c, xf):

        c  = torch.Tensor(c)
        xf = torch.Tensor(xf)
        x0 = torch.zeros(c.shape)

        xf = xf.unsqueeze(0).repeat(len(c),1)
        xr = xf.unsqueeze(1).repeat(1,self.T,1)

        x, u = self.diff_tt_solver(c,x0,xr)

        return c.sum(1).detach().numpy(),  self.rho*( (x[:,-1,:] - xf)**2 ).sum(1).detach().numpy()



    def solve(self, xf):
        cost, pos = self.pyswarms_solver.optimize(self.design_objective, self.pso_iter, xf=xf)
        xf = torch.Tensor(xf).unsqueeze(0)
        xr = xf.unsqueeze(1).repeat(1,self.T,1)
        c  = torch.Tensor( pos ).unsqueeze(0)
        x0 = torch.zeros(c.shape)

        x, u = self.diff_tt_solver(c,x0,xr)

        return c[0], x[0], u[0]



if __name__ == "__main__":

    T = 20
    dt = 0.5

    cmax = 0.32
    cmin = 0.01

    pso_iter = 3
    n_particles = 4


    xf = np.array([0.5,0.7])
    pso_solver = PSOmodel(cmin, cmax, T=T, dt=dt, rho = 1e2, pso_iter = pso_iter,  n_particles = n_particles, lqr_iter = 100)
    c, x, u = pso_solver.solve(xf)

    plt.plot( [xf[0]], [xf[1]],  'r*', label=r"Target point" )
    plt.plot(    x[:,0].detach(),   x[:,1].detach(),  'b*-', label=r"Predicted trajectory, c = [{:.4f},  {:.4f}]".format(c[0], c[1]) )
    plt.xlim(0.0,1.0)
    plt.ylim(0.0,1.0)
    plt.legend()
    plt.show()
