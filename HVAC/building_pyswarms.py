import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
import torch
import matplotlib.pyplot as plt



class PSOmodel():
    def __init__(self, diff_solver, nzones, rho = 1e2, n_particles = 12, pso_iter = 20):
        super().__init__()
        self.pso_iter = pso_iter
        self.n_particles = n_particles
        self.rho    = rho
        self.diff_solver = diff_solver
        self.nzones = nzones
        self.dimensions = 4*nzones

        bounds = (np.zeros(self.dimensions), 10*np.ones(self.dimensions))  # Give a loose upper bound in lieu of None
        options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9}
        self.pyswarms_solver = GlobalBestPSO(n_particles=self.n_particles, dimensions=self.dimensions, bounds=bounds, options=options)



        """
        Calling solve() will populate these lists with the histories of that latest solver call
        """
        self.cost_history = None
        self.mean_pbest_history = None
        self.mean_neighbor_history = None
        self.pos_history = None
        self.velocity_history = None


    def design_objective(self, Bvecs, x0, d, ymin, ymax):

        Bvecs    = torch.Tensor(Bvecs)
        B_list = [ torch.block_diag(*[Bvec[4*i:4*(i+1)].unsqueeze(1) for i in range(self.nzones)]) for Bvec in Bvecs ]
        B = torch.stack(B_list)

        x0   = torch.Tensor(x0).unsqueeze(0).repeat(len(B),1,1)
        d    = torch.Tensor(d).unsqueeze(0).repeat(len(B),1,1)
        ymin = torch.Tensor(ymin).unsqueeze(0).repeat(len(B),1,1)
        ymax = torch.Tensor(ymax).unsqueeze(0).repeat(len(B),1,1)

        solver_out = self.diff_solver(B, x0.squeeze(1), d, ymin, ymax)
        slack_lower = solver_out[3]
        slack_upper = solver_out[4]

        viols_ymin = torch.flatten(slack_lower, start_dim=1)
        viols_ymax = torch.flatten(slack_upper, start_dim=1)

        viol_ymin_sumsq = (viols_ymin**2).sum(1)
        viol_ymax_sumsq = (viols_ymax**2).sum(1)

        viol_norm = torch.norm(torch.cat((viols_ymin, viols_ymax), dim = 1), dim=1, p=2)


        Bsum = torch.flatten(B, start_dim=1).sum(1)
        f = self.rho*(viol_ymin_sumsq + viol_ymax_sumsq) + Bsum

        return f.detach().numpy()



    def design_objective_split(self, Bvecs, x0, d, ymin, ymax):

        Bvecs    = torch.Tensor(Bvecs)
        B_list = [ torch.block_diag(*[Bvec[4*i:4*(i+1)].unsqueeze(1) for i in range(self.nzones)]) for Bvec in Bvecs ]
        B = torch.stack(B_list)

        x0   = torch.Tensor(x0).unsqueeze(0).repeat(len(B),1,1)
        d    = torch.Tensor(d).unsqueeze(0).repeat(len(B),1,1)
        ymin = torch.Tensor(ymin).unsqueeze(0).repeat(len(B),1,1)
        ymax = torch.Tensor(ymax).unsqueeze(0).repeat(len(B),1,1)

        solver_out = self.diff_solver(B, x0.squeeze(1), d, ymin, ymax)
        slack_lower = solver_out[3]
        slack_upper = solver_out[4]

        viols_ymin = torch.flatten(slack_lower, start_dim=1)
        viols_ymax = torch.flatten(slack_upper, start_dim=1)

        viol_ymin_sumsq = (viols_ymin**2).sum(1)
        viol_ymax_sumsq = (viols_ymax**2).sum(1)

        viol_norm = torch.norm(torch.cat((viols_ymin, viols_ymax), dim = 1), dim=1, p=2)

        Bsum = torch.flatten(B, start_dim=1).sum(1)
        f = self.rho*(viol_ymin_sumsq + viol_ymax_sumsq) + Bsum

        return Bsum.detach().numpy(), self.rho*(viol_ymin_sumsq + viol_ymax_sumsq).detach().numpy()


    def solve(self, x0, d, ymin, ymax):

        cost, pos = self.pyswarms_solver.optimize(self.design_objective, self.pso_iter, x0=x0, d=d, ymin=ymin, ymax=ymax)

        Bvecs  = torch.Tensor( pos ).unsqueeze(0)
        B_list = [ torch.block_diag(*[Bvec[4*i:4*(i+1)].unsqueeze(1) for i in range(self.nzones)]) for Bvec in Bvecs ]
        B = torch.stack(B_list)

        x0   = torch.Tensor(x0).unsqueeze(0)
        d    = torch.Tensor(d).unsqueeze(0)
        ymin = torch.Tensor(ymin).unsqueeze(0)
        ymax = torch.Tensor(ymax).unsqueeze(0)

        solver_out = self.diff_solver(B, x0.squeeze(1), d, ymin, ymax)
        slack_lower = solver_out[3]
        slack_upper = solver_out[4]

        viols_ymin = torch.flatten(slack_lower, start_dim=1)
        viols_ymax = torch.flatten(slack_upper, start_dim=1)
        viol_norm = torch.norm(torch.cat((viols_ymin, viols_ymax), dim = 1), dim=1, p=2)

        return B[0], viol_norm[0]



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
