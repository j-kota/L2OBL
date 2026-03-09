import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from twotank_diffmpc import DiffTwoTankSolver
import torch
import matplotlib.pyplot as plt
import os
import pickle
import Nets
from twotank_pyswarms import PSOmodel
from main import c_correction
import time
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--index',                type=int,   default=9999)
args = parser.parse_args()


index = args.index


assert index < 1000

cmax = 0.32
cmin = 0.01
T = 20
dt = 0.5
rho = 1e2

xf_dev = pickle.load( open("xf_dev_standard.p",'rb') )
x0_dev = pickle.load( open("x0_dev_standard.p",'rb') )



"""
Do evaluation of both methods on this xf
"""
(x0,xf) = (x0_dev[index], xf_dev[index])




"""
Instantiate the PSO baseline solver
"""
pso_iter = 200
n_particles = 128

"""
Solve by PSO over the test set
"""

xresid = 9999.9
attempts = 0
start_time = time.time()
while xresid > 0.05 and attempts < 5:
    attempts += 1

    print("\n\n\n\nSolving instance {}, attempt {}".format(index,attempts))
    pso_solver = PSOmodel(cmin, cmax, T=T, dt=dt, rho = rho, pso_iter = pso_iter,  n_particles = n_particles, lqr_iter = 200)
    c, x, u = pso_solver.solve(xf)

    xresid = torch.norm( x[-1] - xf ).mean().item()

end_time = time.time()
solve_time = end_time - start_time

cost_history = pso_solver.pyswarms_solver.cost_history
pos_history  = pso_solver.pyswarms_solver.pos_history


pos_tr = torch.stack([torch.Tensor(c) for c in pos_history])
pos_tr_flat = pos_tr.view(-1,2)


pair_list = [pso_solver.design_objective_split(c, xf) for c in pos_tr]
cost_history_all = torch.Tensor([a+b for (a,b) in pair_list])
obj_history_all = torch.Tensor([a for (a,b) in pair_list])
pen_history_all = torch.Tensor([b for (a,b) in pair_list])

best_cost_indices = torch.tensor(  [torch.argmin( cost_history_all[:(k+1)].flatten() ) for k in range(len(pos_tr))]   )  # best index of the flattened cost vector, per time step

cost_history_recon_best = [cost_history_all[:(k+1)].flatten()[best_cost_indices[k]].item() for k in range(len(pos_tr))]
obj_history_recon_best  = [obj_history_all[:(k+1)].flatten()[best_cost_indices[k]].item() for k in range(len(pos_tr))]
pen_history_recon_best  = [pen_history_all[:(k+1)].flatten()[best_cost_indices[k]].item() for k in range(len(pos_tr))]

cost_history_recon  =  [torch.Tensor(costs).min() for costs in cost_history_all]
cost_history_recon_best2 = [ min(cost_history_recon[:(k+1)]).item() for k in range(len(cost_history_recon)) ]

pso_out = {}
pso_out["index"] = index
pso_out["xf"] = xf
pso_out["c"] = c
pso_out["xresid"] = xresid
pso_out["cost_history"] = cost_history
pso_out["obj_history_recon_best"]  = obj_history_recon_best
pso_out["pen_history_recon_best"]  = pen_history_recon_best
pso_out["cost_history_recon_best"] = cost_history_recon_best
pso_out["pos_history"] = pos_history
pso_out["solve_time"] = solve_time
pso_out["attempts"] = attempts
outfile_name = "pso_out_{}.p".format(index)
path_name = os.path.join("./PSO_solves/", outfile_name)
pickle.dump(pso_out, open(path_name,'wb'))
