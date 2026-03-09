import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import cvxpy as cv

import matplotlib.pyplot as plt
from QP_layer import get_QP_layer
import scipy

relu = torch.nn.ReLU()


def x_correction(x, A, b, E, diff_solver, alpha, n_corr_steps = 5):

        viols_list = []
        for _ in range(n_corr_steps):
            z = diff_solver(x)[0]
            viols = relu( x@A.T - (b+z@E.T) )
            grad_viol = 2*(viols)

            viols_list.append( viols.mean().item() )

            grads_x = torch.autograd.grad( viols, x, grad_viol, retain_graph = True )[0]
            x = x - alpha*grads_x

        return x, viols_list



if __name__ == "__main__":

    data = scipy.io.loadmat('./param_sol_data.mat')
    print("data")
    print( data )

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

    diff_solver = get_QP_layer(nx,nz,nc,nd,H,e,f,F,h,G)

    xinit = torch.rand(10,nx)

    alpha = 0.1
    n_corr_steps = 5
    xinit.requires_grad = True
    x, viols_list = x_correction(xinit, A, b, E, alpha, n_corr_steps)
    plt.semilogy(range(len(viols_list)), viols_list, label = "violations")
    plt.legend()
    plt.show()
