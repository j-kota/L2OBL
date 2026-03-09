
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import Nets
import matplotlib.pyplot as plt
from torch.func import functional_call, vmap, grad
from twotank_diffmpc import DiffTwoTankSolver
import argparse
import pickle
import copy
import time

parser = argparse.ArgumentParser()

parser.add_argument('--n_train',                type=int,   default=3000)
parser.add_argument('--n_dev',                  type=int,   default= 1000)
parser.add_argument('--lqr_iter_train',         type=int,   default= 50)
parser.add_argument('--lqr_iter_test',          type=int,   default= 100)
parser.add_argument('--lr',                     type=float,   default=5e-4)

parser.add_argument('--penalty',     type=float,   default=1.0)

parser.add_argument('--alpha',       type=float,   default=0.001)   # correction step size
parser.add_argument('--n_corr_steps', type=int,     default=3)

parser.add_argument('--epochs',       type=int,     default=30)
parser.add_argument('--index',       type=int,     default=9999)
args = parser.parse_args()



def compute_traj(c, x0, xr, cl_system):

    x,u = cl_system(c,x0,xr)
    return x,u


def c_correction(c, x0, xr, cl_system, c_restore, alpha, n_corr_steps):

    xresid_list = []
    for _ in range(n_corr_steps):

        c = c_restore(c)

        x, u = compute_traj(c, x0, xr, cl_system)
        x_f  = x[:,-1,:]
        xr_f = xr[:,-1,:]

        grads_x = 2*(x_f - xr_f)

        xresid = torch.norm( x_f - xr_f , dim=1).mean().item()
        xresid_list.append(xresid)

        grads_c = torch.autograd.grad( x_f, c, grads_x, retain_graph = True )[0]
        c = c - alpha*grads_c

    c = c_restore(c)

    return c




if __name__ == "__main__":

    seed = 2
    torch.manual_seed(seed)
    np.random.seed(seed)
    plotting = False


    """
    Instantiate the two-tank control system class and variables
    """
    nx = 2
    nu = 2
    nref = nx
    nc = 2

    nsteps = 20                 # prediction horizon
    ntraj = nx*(nsteps + 1)     # size of the reference trajectory (for NN input)
                                #    (trajectory includes nsteps+1 for initial condition)


    """
    Initiate an upper-level prediction model, and generate its initial
       distribution over upper-level variables c, for DPC training
    """

    c_input_size = nc
    c_net = Nets.ReLUnet(c_input_size, 1, hidden_sizes = [(c_input_size), 5*(c_input_size), 30*(c_input_size), 50*(c_input_size), 30*(c_input_size), 5*(c_input_size), (c_input_size)], batch_norm = True, initialize = True)
    sigmoid = torch.nn.Sigmoid()




    """
    Instantiate some elements of the dataset: static start and end points for the control task
    """

    n_train = args.n_train
    n_dev   = args.n_dev

    x0 = torch.Tensor([0.0,0.0])

    # Generate train x0 and xr
    x0_train = x0*torch.ones(n_train, 2)
    if True:
        x2min = 0.2
        x1minrat = 0.5
        x2 = (1.0-x2min)*torch.rand(n_train,1)+x2min;   x1 = x2*(  (1.0-x1minrat)*torch.rand(n_train,1) + x1minrat  )
    xf_train = torch.cat((x1,x2),dim=1)
    xr_train = xf_train.unsqueeze(1).repeat(1,nsteps+1,1)

    xf_dev = pickle.load( open("xf_dev_standard.p",'rb') )  # replaces the above lines
    x0_dev = x0*torch.ones(len(xf_dev), 2)
    xr_dev = xf_dev.unsqueeze(1).repeat(1,nsteps+1,1)


    """ Create and pre-train the inner-loop DPC model """
    umin = 0
    umax = 1.
    xmin = 0
    xmax = 1.


    """  Define dataset for c: used for DPC training only!!  """
    cmax = 0.32
    cmin = 0.01


    def c_sigmoid(c_in):
        return ( cmin + (cmax-cmin)*sigmoid(c_in) )

    def c_project(c_in):
        return torch.clamp(c_in, min=cmin, max=cmax)


    T = nsteps+1
    dt = 0.5
    print("dt = ")
    print( dt    )
    cl_system_train = DiffTwoTankSolver(T, dt, lqr_iter = args.lqr_iter_train, verbose = 1)  # TODO: target_system is a misnomer, call it cl_system or something
    cl_system_test  = DiffTwoTankSolver(T, dt, lqr_iter = args.lqr_iter_test,  verbose = 1)

    """
     Create the upper-level training dataset
    """

    batch_size_ul = 32
    c_optimizer = torch.optim.Adam(c_net.parameters(), lr=args.lr)
    c_train_loader = DataLoader( list(zip(x0_train, xf_train)), shuffle=True, batch_size=batch_size_ul )
    c_test_loader  = DataLoader( list(zip(x0_dev,   xf_dev)),   shuffle=True, batch_size=50 ) # Batch size should always be a factor of the test set size


    alpha = args.alpha
    corr_steps_train = args.n_corr_steps
    corr_steps_test  = 10
    c_restore_train = lambda c, x0, xr: c_correction(c, x0, xr, cl_system_train, c_project, alpha, corr_steps_train)
    c_restore_test  = lambda c, x0, xr: c_correction(c, x0, xr, cl_system_test,  c_project, alpha, corr_steps_test)


    """
    Upper-level training loop
    """
    train_xloss_list = []
    train_csum_list = []
    test_xloss_list = []
    test_csum_list = []

    test_loss_list = []
    test_ref_mean_list = []
    test_cdiff_list = []
    test_cabs_list = []
    test_xresid_list = []
    test_cresid_list = []

    # Standard deviation
    test_csum_std_list = []
    test_xresid_std_list = []

    for epoch in range(args.epochs+1):


        """Upper-level eval routine"""
        batch_csum_list = []
        batch_xresid_list = []
        batch_xloss_list = []
        for (x0, xf) in c_test_loader:
            print("New Test Iter")

            batsize = len(xf)
            xr = xf.unsqueeze(1).repeat(1,nsteps+1,1)

            pred_input = xf
            c = c_net(pred_input)
            c = c_sigmoid(c)
            start_time = time.time()
            c = c_restore_test( c, x0, xr )
            end_time = time.time()
            print("Solve time = {}".format( (end_time-start_time)/len(xf)   ))
            x,u = cl_system_test(c,x0,xr)

            cdiff = (c[:,0] - c[:,1]).mean()
            cabs  = (c[:,0] - c[:,1]).abs().mean()
            xresid = torch.norm( x[:,-1,:] - xr[:,-1,:] , dim=1)

            xloss = (   ( x[:,-1,:] - xr[:,-1,:] )**2   ).sum(1)
            csum  = c.sum(1)

            x_dev = x  # save for plotting at the end
            if epoch==0:
                x_dev_init = x



            batch_csum_list.append(csum)
            batch_xresid_list.append(xresid)
            batch_xloss_list.append(xloss)


        test_csum_avg   = torch.cat( tuple(batch_csum_list), dim=0 ).mean().item()
        test_xresid_avg = torch.cat( tuple(batch_xresid_list), dim=0 ).mean().item()
        test_csum_std   = torch.cat( tuple(batch_csum_list), dim=0 ).std().item()
        test_xresid_std = torch.cat( tuple(batch_xresid_list), dim=0 ).std().item()
        test_xloss_avg  = torch.cat( tuple(batch_xloss_list), dim=0 ).mean().item()

        test_csum_list.append(  test_csum_avg    )
        test_xresid_list.append( test_xresid_avg )
        test_csum_std_list.append(  test_csum_std    )
        test_xresid_std_list.append( test_xresid_std )
        test_xloss_list.append( test_xloss_avg )

        print("Epoch {}".format(epoch))
        print("xresid = {}".format(test_xresid_avg))
        print("csum = {}".format(test_csum_avg))

        if epoch == args.epochs:    # Skip training in the last epoch
            break;


        """ Upper-level training of 1 epoch """
        # Training: predict c given x0,xr
        for (x0, xf) in c_train_loader:
            if torch.rand(1).item() < 0.5: continue
            print("New Training Iter")
            batsize = len(x0)
            xr = xf.unsqueeze(1).repeat(1,nsteps+1,1)

            pred_input = xf
            c = c_net(pred_input)
            c = c_sigmoid(c)
            c = c_restore_train( c, x0, xr )

            x,u = cl_system_train(c,x0,xr)

            xloss = torch.nn.MSELoss()( x[:,-1,:], xr[:,-1,:] )
            csum = c.sum(1).mean()
            closs = 1.0*csum
            cdiff = (c[:,0] - c[:,1]).mean()

            loss = args.penalty*xloss + closs
            loss.backward()
            c_optimizer.step()
            c_optimizer.zero_grad()

            train_xloss_list.append(xloss.detach().mean().item())
            train_csum_list.append(csum.detach().mean().item())


    train_png_name = "./plt/tt_diffmpc_traincurves"
    for (k,v) in vars(args).items():
        train_png_name += "__" + k + "-" + str(v)
    train_png_name += ".pdf"
    plt.semilogy( range(len(train_xloss_list)), train_xloss_list, label = 'xloss')
    plt.semilogy( range(len(train_csum_list)),  train_csum_list,  label = 'csum')
    plt.xlabel('Outer training iteration')
    plt.ylabel('Training set batch loss')
    plt.legend()
    #plt.show()
    plt.savefig(train_png_name)
    plt.clf()

    test_png_name = "./plt/tt_diffmpc_testcurves"
    for (k,v) in vars(args).items():
        test_png_name += "__" + k + "-" + str(v)
    test_png_name += ".pdf"


    # Compute std fill ranges for xresid and csum
    test_xresid_avg_np = np.array(test_xresid_list)
    test_xresid_std_np = np.array(test_xresid_std_list)
    test_xresid_up   = (test_xresid_avg_np + test_xresid_std_np).tolist()
    test_xresid_dwn  = (test_xresid_avg_np - test_xresid_std_np).tolist()

    test_csum_avg_np = np.array(test_csum_list)
    test_csum_std_np = np.array(test_csum_std_list)
    test_csum_up   = (test_csum_avg_np + test_csum_std_np).tolist()
    test_csum_dwn  = (test_csum_avg_np - test_csum_std_np).tolist()


    plt.semilogy( range(len(test_xloss_list)), test_xloss_list, label = r"$\| x_{N}-r \|^2")
    plt.semilogy( range(len(test_xresid_list)), test_xresid_list, label = r"$\| x_{N}-r \|")
    plt.semilogy( range(len(test_csum_list)),  test_csum_list,  label = r"$\Sigma c")

    plt.fill_between(range(len(test_xresid_list)), test_xresid_dwn, test_xresid_up, color = 'lightskyblue')
    plt.fill_between(range(len(test_csum_list)), test_csum_dwn, test_csum_up, color = 'yellow')

    plt.xlabel('Outer test epoch')
    plt.ylabel('Test set loss')
    plt.legend()
    plt.savefig(test_png_name)
    plt.clf()

    # Save the trained model
    state_dict_name = "./models/tt_diffmpc_statedict"
    for (k,v) in vars(args).items():
        state_dict_name += "__" + k + "-" + str(v)
    state_dict_name += ".pth"
    torch.save(c_net.state_dict(), state_dict_name)


    # Save to pickle: the model, training curves and test/dev set
    output_dict_name = "./pickle/tt_diffmpc_outdict"
    for (k,v) in vars(args).items():
        output_dict_name += "__" + k + "-" + str(v)
    output_dict_name += ".p"

    output_dict = copy.deepcopy( vars(args) )
    output_dict["train_xloss_list"] = train_xloss_list
    output_dict["train_csum_list"]  = train_csum_list
    output_dict["test_xloss_list"]  = test_xloss_list
    output_dict["test_xresid_list"] = test_xresid_list
    output_dict["test_csum_list"]   = test_csum_list
    output_dict["test_csum_std_list"]   = test_csum_std_list
    output_dict["test_xresid_std_list"] = test_xresid_std_list
    output_dict["x0_dev"]           = x0_dev
    output_dict["xf_dev"]           = xf_dev
    output_dict["cmax"]             = cmax
    output_dict["cmin"]             = cmin
    pickle.dump(output_dict, open(output_dict_name,'wb'))
