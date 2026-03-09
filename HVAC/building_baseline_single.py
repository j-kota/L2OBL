
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from matplotlib.lines import Line2D

import neuromancer.psl as psl
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.plot import pltCL

from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import cvxpy as cv
from building_MPC_layer import get_building_MPC_layer, get_building_MPC_layer
import os
import building_utils
import pickle
import argparse

from pyswarms.single.global_best import GlobalBestPSO
import torch
import matplotlib.pyplot as plt
from building_pyswarms import PSOmodel
import time

parser = argparse.ArgumentParser()
parser.add_argument('--index',      type=int,   default=9999)
parser.add_argument('--nsteps',     type=int,     default=30)
parser.add_argument('--nzones',     type=int,     default=1)
parser.add_argument('--linkzones',  type=bool,    default=False)
parser.add_argument('--upos',       type=bool,    default=False)
parser.add_argument('--ntrain',     type=int,     default=10000)
parser.add_argument('--ntest',      type=int,     default=1000)
args = parser.parse_args()

index = args.index
assert index < 1000

"""
Get the solver and dataset
"""

torch.manual_seed(0)
np.random.seed(0)

# ground truth system model
sys = psl.systems['LinearSimpleSingleZone'](seed=0)
nzones = args.nzones

# problem dimensions
nx = nzones*sys.nx                # number of states
nu = nzones*sys.nu                # number of control inputs
nd = nzones*sys.nD                # number of disturbances
ny = nzones*sys.ny                # number of controlled outputs
nref = ny                  # number of references
partial_observe = False
if partial_observe:        # Toggle partial observability of the disturbance
    d_idx = sys.d_idx
else:
    d_idx = range(nd)
nd_obs = len(d_idx)
nB = nu*nx
# extract exact state space model matrices:
A = torch.block_diag(  *tuple([torch.tensor(sys.A)    for _ in range(nzones)])  )
B = torch.block_diag(  *tuple([torch.tensor(sys.Beta) for _ in range(nzones)])  )
C = torch.block_diag(  *tuple([torch.tensor(sys.C)    for _ in range(nzones)])  )
E = torch.block_diag(  *tuple([torch.tensor(sys.E)    for _ in range(nzones)])  )
F = torch.zeros(ny)
G = torch.zeros(nx)
y_ss = torch.zeros(ny)

dx = 0.025
nx_zone = 4
if nzones > 1:
    A = building_utils.link_zones_long(A, nzones, nx_zone, dx)

umax = torch.Tensor([sys.umax.item() for _ in range(nzones)])

B_scale_factor = umax.max()
umax = umax / B_scale_factor
B = B * B_scale_factor

umin = torch.Tensor([sys.umin.item() for _ in range(nzones)]) if args.upos else -umax


nsteps = args.nsteps
ntrain = args.ntrain
ntest  = args.ntest
batch_size_ul = 100

n_samples = ntrain+ntest

# generate data for a single-zone building
filename = "./data/zone_data_{}steps_{}samples.p".format(nsteps, n_samples)
zone_data = pickle.load(open(filename,'rb'))
#if not os.path.isfile(filename):
#    zone_data = building_utils.gen_building_data_single(n_samples, nsteps)
#    pickle.dump(zone_data,open(filename,'wb'))
#else:
#    zone_data = pickle.load(open(filename,'rb'))


batched_ymin, batched_ymax, batched_dist, batched_x0 = zone_data

# duplicate data across multiple zones
batched_x0   = batched_x0.repeat(1,1,nzones)
batched_y0   = batched_x0[:,:,[-1]].repeat(1,1,nzones)
batched_ymin = batched_ymin.repeat(1,1,nzones)
batched_ymax = batched_ymax.repeat(1,1,nzones)
batched_dist = batched_dist.repeat(1,1,nzones)

Q_weight = 50.0
R_weight =  1.0
diff_solver = get_building_MPC_layer(nsteps,nu,nx,ny,nd, umin,umax, A,C,E,F,G,y_ss, Q_weight,R_weight)



"""
Train/test split   (only using Test in this file)
"""
train_ymin = batched_ymin[:ntrain]
train_ymax = batched_ymax[:ntrain]
train_dist = batched_dist[:ntrain]
train_x0   =   batched_x0[:ntrain]

test_ymin = batched_ymin[-ntest:]
test_ymax = batched_ymax[-ntest:]
test_dist = batched_dist[-ntest:]
test_x0   =   batched_x0[-ntest:]

ymin = test_ymin[index]
ymax = test_ymax[index]
dist = test_dist[index]
x0   = test_x0[index]



"""
Instantiate the PSO baseline solver
"""
pso_iter = 200
n_particles = 128
rho = 5.0

"""
Solve by PSO over the test set
"""

xresid = 9999.9
attempts = 0
start_time = time.time()

print("\n\n\n\nSolving instance {}, attempt {}".format(index,attempts))
pso_solver = PSOmodel(diff_solver, nzones, rho = rho, pso_iter = pso_iter,  n_particles = n_particles)
B, xresid = pso_solver.solve(x0, dist, ymin, ymax)

end_time = time.time()
solve_time = end_time - start_time

cost_history = pso_solver.pyswarms_solver.cost_history
pos_history  = pso_solver.pyswarms_solver.pos_history

pos_tr = torch.stack([torch.Tensor(c) for c in pos_history])
pos_tr_flat = pos_tr.view(-1,2)   # 2 is the dimension of pos



pair_list = [pso_solver.design_objective_split(c, x0, dist, ymin, ymax) for c in pos_tr]
cost_history_all = torch.Tensor([a+b for (a,b) in pair_list])
obj_history_all = torch.Tensor([a for (a,b) in pair_list])
pen_history_all = torch.Tensor([b for (a,b) in pair_list])

best_cost_indices = torch.tensor(  [torch.argmin( cost_history_all[:(k+1)].flatten() ) for k in range(len(pos_tr))]   )  # best index of the flattened cost vector, per time step

cost_history_recon_best = [cost_history_all[:(k+1)].flatten()[best_cost_indices[k]].item() for k in range(len(pos_tr))]
obj_history_recon_best  = [obj_history_all[:(k+1)].flatten()[best_cost_indices[k]].item() for k in range(len(pos_tr))]
pen_history_recon_best  = [pen_history_all[:(k+1)].flatten()[best_cost_indices[k]].item() for k in range(len(pos_tr))]



pso_out = {}
pso_out["index"] = index
pso_out["xresid"] = xresid

pso_out["obj_history_recon_best"]  = obj_history_recon_best
pso_out["pen_history_recon_best"]  = pen_history_recon_best
pso_out["cost_history_recon_best"] = cost_history_recon_best

pso_out["solve_time"] = solve_time
pso_out["attempts"] = attempts
outfile_name = "pso_out_{}.p".format(index)
path_name = os.path.join("./PSO_solves/", outfile_name)
pickle.dump(pso_out, open(path_name,'wb'))
