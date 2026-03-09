import cvxpy as cp
from cvxpy import *
import numpy as np
import scipy as sp

from scipy import sparse
from pylab import *
import time

from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import torch





def get_QP_layer(nx,nz,nc,nd,H,e,f,F,h,G):

    z = Variable(nz)
    x = Parameter(nx)

    e = e.flatten()
    f = f.flatten()
    h = h.flatten()

    objective = quad_form(z,0.5*H) + e@z + f@x
    constraints  = [F@z <= h + G@x]

    prob = Problem(Minimize(objective), constraints)

    return CvxpyLayer(prob, parameters=[x], variables=[z])
