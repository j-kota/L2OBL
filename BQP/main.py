import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.lines import Line2D
import cvxpy as cv
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import matplotlib.pyplot as plt
import Nets
import os
import pickle
import argparse
import scipy.io
import pickle
from QP_layer import get_QP_layer
from QP_correction import x_correction
import time

parser = argparse.ArgumentParser()
parser.add_argument('--lr',          type=float,   default=1e-1)
parser.add_argument('--penalty',     type=float,   default=100.0)
parser.add_argument('--alpha',       type=float,   default=1e-3)
parser.add_argument('--index',       type=int,     default=9999)

parser.add_argument('--n_corr_steps', type=int,     default=10)
parser.add_argument('--width_factor', type=int,     default=10)
parser.add_argument('--input_file',   type=str,     default="param_sol_data_3_2_10000.mat")
parser.add_argument('--epochs',       type=int,     default=30)
args = parser.parse_args()

input_data_path = "./"+args.input_file

data = scipy.io.loadmat(input_data_path)


cstack = torch.Tensor(data["cstack"])
dstack = torch.Tensor(data["dstack"])
Q      = torch.Tensor(data["Q"])
A      = torch.Tensor(data["A"])
b      = torch.Tensor(data["b"]).flatten()
E      = torch.Tensor(data["E"])
H      = torch.Tensor(data["H"])
e      = torch.Tensor(data["e"]).flatten()
f      = torch.Tensor(data["f"]).flatten()
F      = torch.Tensor(data["F"])
h      = torch.Tensor(data["h"]).flatten()
G      = torch.Tensor(data["G"])
xstack = torch.Tensor(data["xstack"])
zstack = torch.Tensor(data["zstack"])


# problem dimensions
nx = xstack.shape[1]
nz = zstack.shape[1]
nc = cstack.shape[1]
nd = dstack.shape[1]

nsamples = dstack.shape[0]
batch_size_ul = 100


"""
Train/test split
"""
ntrain = 0#int( nsamples*0.8 )
ntest  = 1000

xtest  = xstack[:ntest]
ztest  = zstack[:ntest]
ctest  = cstack[:ntest]
dtest  = dstack[:ntest]






"""
Upper-level prediction model components
"""
relu = torch.nn.ReLU()
sigmoid = torch.nn.Sigmoid()


input_size  = nc + nd
output_size = nx
x_predictor = Nets.ReLUnet(input_size, output_size, hidden_sizes = [(input_size),5*args.width_factor*(input_size),10*args.width_factor*(input_size), 20*args.width_factor*(input_size), 10*args.width_factor*(input_size), 5*args.width_factor*(input_size), (input_size), output_size], batch_norm = True, initialize = True)

diff_solver = get_QP_layer(nx,nz,nc,nd,H,e,f,F,h,G)


"""
Create the upper-level training dataset
"""
optimizer = torch.optim.Adam(x_predictor.parameters(), lr=args.lr)
test_loader  = DataLoader( list(zip( ctest,  dtest,  xtest,  ztest)), shuffle=False, batch_size=batch_size_ul )


def ul_objective(x,z,c,d):
    return (   0.5*(x*(x@Q.T)).sum(1) + (c*x).sum(1) + (d*z).sum(1) + 100.0   )

penalty_weight = args.penalty
epochs = args.epochs

alpha = args.alpha

train_loss_list = []
train_obj_list  = []
train_viols_sumsq_list  = []

test_loss_list = []
test_obj_list  = []
test_objstd_list  = []
test_objgap_list  = []
test_objgapstd_list  = []
test_viols_sumsq_list  = []
test_viols_norm_list  = []
test_viols_norm_std_list  = []
test_viols_mean_list  = []
test_viols_mean_std_list  = []

test_xresid_list  = []
test_xresid_std_list  = []
test_solve_time = None
for epoch in range(epochs):
    print("Epoch {}".format(epoch))

    if True:
        (c,d,xopt,zopt) = (ctest,dtest,xtest,ztest)
        x = x_predictor( torch.cat((c,d),dim=1) )
        start_time = time.time()
        x, viols_list = x_correction(x, A, b, E, diff_solver, alpha, n_corr_steps=20)
        end_time = time.time()
        test_solve_time = (end_time - start_time)/len(x)
        z = diff_solver(x)[0]

        viols = relu( x@A.T - (b+z@E.T) )
        viols_sumsq = (viols**2).sum(1).mean()

        viols_norm  = torch.norm(viols, dim=1, p=2).mean()
        viols_mean  = viols.mean(1).mean()

        viols_norm_std  = torch.norm(viols, dim=1, p=2).std()
        viols_mean_std  = viols.mean(1).std()

        objopt  = ul_objective(xopt,zopt,c,d).mean()
        obj     = ul_objective(x,z,c,d).mean()
        objstd  = ul_objective(x,z,c,d).std()
        objgap    = ( (ul_objective(x,z,c,d) - ul_objective(xopt,zopt,c,d))/ul_objective(xopt,zopt,c,d) ).mean()
        objgapstd = ( (ul_objective(x,z,c,d) - ul_objective(xopt,zopt,c,d))/ul_objective(xopt,zopt,c,d) ).std()

        xresid = torch.norm(x - xopt, dim=1, p=2).mean()
        xresid_std = torch.norm(x - xopt, dim=1, p=2).std()


        loss = penalty_weight*(viols_sumsq) + obj

        test_loss_list.append(loss.item())
        test_obj_list.append(obj.item())
        test_objstd_list.append(objstd.item())
        test_objgap_list.append(objgap.item())
        test_objgapstd_list.append(objgapstd.item())
        test_viols_sumsq_list.append(viols_sumsq.item())
        test_viols_norm_list.append(viols_norm.item())
        test_viols_norm_std_list.append(viols_norm_std.item())
        test_viols_mean_list.append(viols_mean.item())
        test_viols_mean_std_list.append(viols_mean_std.item())
        test_xresid_list.append(xresid.item())
        test_xresid_std_list.append(xresid_std.item())

        print("Optimal Objective = {}".format(objopt.item()))
        print("Objective gap = {}".format(objgap.item()))
        print("Coupling violation = {}".format(viols_norm.item()))


    for _ in range(400):
        print("Iteration {}".format(_))
        c = 2*torch.Tensor(np.random.rand(batch_size_ul, nc));
        d = 2*torch.Tensor(np.random.rand(batch_size_ul, nd));

        x = x_predictor( torch.cat((c,d),dim=1) )
        x, viols_list = x_correction(x, A, b, E, diff_solver, alpha, n_corr_steps=args.n_corr_steps)
        z = diff_solver(x)[0]

        viols = relu( x@A.T - (b+z@E.T) )
        viols_sumsq = (viols**2).sum(1).mean()

        obj = ul_objective(x,z,c,d).mean()

        loss = penalty_weight*(viols_sumsq) + obj

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss_list.append(loss.item())
        train_obj_list.append(obj.item())
        train_viols_sumsq_list.append(viols_sumsq.item())








loss_png_name = "QP_loss_curves"
for (k,v) in vars(args).items():
    loss_png_name += "__" + k + "-" + str(v)
loss_png_name += ".png"

fig, axs = plt.subplots(5,1,figsize=(9,12))
axs[0].tick_params(axis='y', labelsize=12)
axs[1].tick_params(axis='y', labelsize=12)
axs[2].tick_params(axis='y', labelsize=12)
axs[3].tick_params(axis='y', labelsize=12)
axs[4].tick_params(axis='y', labelsize=12)

axs[0].semilogy(range(len(test_viols_norm_list)), test_viols_norm_list, 'k-', label=r"$\|viol\|_2$")
axs[0].semilogy(range(len(test_viols_mean_list)), test_viols_mean_list, 'k-', label=r"$\frac{1}{N}\sum viol$")
axs[0].legend(fontsize=14)

test_objgap_up   = (np.array(test_objgap_list) + np.array(test_objgapstd_list)).tolist()
test_objgap_dwn  = (np.array(test_objgap_list) - np.array(test_objgapstd_list)).tolist()
axs[1].plot(range(len(test_objgap_list)), test_objgap_list, 'b-', label=r"Objective Gap")
axs[1].fill_between(range(len(test_objgap_list)), test_objgap_dwn, test_objgap_up, color = 'lightskyblue')
axs[1].set_ylabel('Upper-Level Objective: Test Set')
axs[1].legend(fontsize=14)

test_objgap_up   = (np.array(test_objgap_list) + np.array(test_objgapstd_list)).tolist()
test_objgap_dwn  = (np.array(test_objgap_list) - np.array(test_objgapstd_list)).tolist()
axs[2].semilogy(range(len(test_objgap_list)), test_objgap_list, 'b-', label=r"Objective Gap")
axs[2].fill_between(range(len(test_objgap_list)), test_objgap_dwn, test_objgap_up, color = 'lightskyblue')
axs[2].set_ylabel('Upper-Level Objective: Test Set')
axs[2].legend(fontsize=14)

xresid_up   = (np.array(test_xresid_list) + np.array(test_xresid_std_list)).tolist()
xresid_dwn  = (np.array(test_xresid_list) - np.array(test_xresid_std_list)).tolist()
axs[3].semilogy(range(len(test_xresid_list)), test_xresid_list, 'b-', label=r"\| x - x_{opt}  \|_2")
axs[3].fill_between(range(len(test_xresid_list)), xresid_dwn, xresid_up, color = 'lightskyblue')
axs[3].set_ylabel('Upper-Level Solution Residual: Test Set')
axs[3].legend(fontsize=14)

axs[4].plot(range(len(test_loss_list)), test_loss_list, label = "Loss")
axs[4].set_ylabel('Loss Function: Test Set')
axs[4].set_xlabel('Training Epoch')
axs[4].legend(fontsize=14)
plt.savefig("./plt/QP_test_"+loss_png_name)



output_dict_name = "./pickle/QP_outdict"
for (k,v) in vars(args).items():
    output_dict_name += "__" + k + "-" + str(v)
output_dict_name += ".p"

output_dict = vars(args)
output_dict["objopt"] = objopt.item()
output_dict["test_obj_list"] = test_obj_list
output_dict["test_objgap_list"] = test_objgap_list
output_dict["test_objgapstd_list"] = test_objgapstd_list
output_dict["test_loss_list"] = test_loss_list
output_dict["test_viols_norm_list"] = test_viols_norm_list
output_dict["test_viols_norm_std_list"] = test_viols_norm_std_list
output_dict["test_viols_mean_list"] = test_viols_mean_list
output_dict["test_viols_mean_std_list"] = test_viols_mean_std_list
output_dict["test_xresid_list"] = test_xresid_list
output_dict["test_xresid_std_list"] = test_xresid_std_list
output_dict["test_solve_time"] = test_solve_time

pickle.dump(output_dict,open(output_dict_name,'wb'))
