#!/usr/bin/env python
# coding: utf-8

"""
---------------------------------------------------------------------------------------
Physics Informed Neural Networks (PINNs) -2D Wave Equation - PyTorch
---------------------------------------------------------------------------------------
Training Neural Network to converge towards a well-defined solution of a PDE by way of
 minimising for the residuals across the spatio-temporal domain.
Initial and Boundary conditions are met by introducing them into the loss function along
 with the PDE residuals.
Equation:
-----------------------------------------------------------------------------------------
u_tt = u_xx + u_yy on [-1,1] x [-1,1]
Dirichlet Boundary Conditions :
u=0
#
Initial Distribution :
u(x,y,t=0) = exp(-40(x-4)^2 + y^2)
Initial Velocity Condition :
u_t(x,y,t=0) = 0
m*N layers for mth order PDE
-----------------------------------------
Parameter changes to play with:
--------------------------------------------------------------------------------------
CommandLineArgs class gives 7 parameters that can be changed to edit performance
3 are sample sizes for training
3 are domain specific
1 is for training loops
Note
-------------------------------------------------------------------------------------
Building the numerical solution by solving the Wave Equation using a spectral solver
implemented on numpy.
Numerical Method - Spectral Solver using FFT with solution code from Boston University
with their permission
The numerical solution will not form the training data but will be used for comparing
 against the PINN solution.
"""

import os
import sys
import time
import operator
from functools import reduce
import argparse

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from pyDOE import lhs
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
__author__ = "Lucy Harris, Vignesh Gopakumar"
__license__ = "GPL 2"
__email__ = "lucy.harris@ukaea.uk"


class CommandLineArgs:
    """
    Take arguments from command line
    Parameters:
    ---------------------------------------------------------
    Epochs                 (-E, --epochs,    default=20000)
    Inital Sampling No.    (-I, --inital,    default=1000)
    Boundary Sampling No.  (-B, --boundary,  default=1000)
    Domain Sampling No.    (-D, --domain,    default=20000)
    Spatial Discretisation (-N, --n-steps,   default=50)
    Simulation Time (s)    (-T, --time,      default=1)
    Grid size              (-G, --grid-size, default=50)
    """

    def __init__(self):
        pinn_parser = argparse.ArgumentParser(
            description="PINN solver for Schrödinger's wave equation",
            fromfile_prefix_chars="@",
            allow_abbrev=False,
            epilog="Enjoy the program! :)",
        )
        pinn_parser.add_argument(
            "-E",
            "--epochs",
            action="store",
            type=int,
            default=30000,
            help="set number of epochs for training",
        )

        pinn_parser.add_argument(
            "-I",
            "--initial",
            action="store",
            type=int,
            default=128*128,
            help="set no. initial samples N_i",
        )

        pinn_parser.add_argument(
            "-B",
            "--boundary",
            action="store",
            type=int,
            default=5000,
            help="set no. boundary samples N_b",
        )

        pinn_parser.add_argument(
            "-D",
            "--domain",
            action="store",
            type=int,
            default=50000,
            help="set no. domain samples N_f",
        )
        pinn_parser.add_argument(
            "-N",
            "--n-steps",
            action="store",
            type=int,
            default=30,
            help="set spatial discretisation of domain",
        )

        pinn_parser.add_argument(
            "-T",
            "--time",
            action="store",
            type=int,
            default=64*float(float(1/25)),
            help="set simulation time of domain",
        )

        pinn_parser.add_argument(
            "-G",
            "--grid-size",
            action="store",
            type=int,
            default=128,
            help="set grid size for domain",
        )

        self.args = pinn_parser.parse_args()
        print("Command grid_sizeine Arguments: ", vars(self.args))
        sys.stdout.flush()


class TorchConfig:
    def __init__(self):
        self.default_device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        torch.set_default_dtype(dtype)

    def torch_tensor_grad(self, x, device):
        if device == 'cuda':
            x = torch.cuda.FloatTensor(x)
        else:
            x = torch.FloatTensor(x)
        x.requires_grad = True
        return x

    def torch_tensor_nograd(self, x, device):
        if device == 'cuda':
            x = torch.cuda.FloatTensor(x)
        else:
            x = torch.FloatTensor(x)
        x.requires_grad = False
        return x


class WaveEquation:
    """
    Numerical method for 2D wave equation with spectral solver using FFT.
    Code from Boston University with permission
    Parameters:
    -----------------------------------------------------------
    spatial_discretisation (int) : size of discrete resolution in domain
    simulation_time (int) : number of seconds of simulation
    grid_size (int) : full size of sample grid domain
    Return:
    -----------------------------------------------------------
    numerical solution of output u (array)
    """

    def __init__(self,  simulation_time, grid_size):
        self.simulation_time = simulation_time
        self.grid_size = grid_size
        self.x0 = 0.0
        self.xf = 128.0
        self.y0 = 0.0
        self.yf = 128.0
        self.initialization()
        self.initCond()

    def initialization(self):
        self.x = np.linspace(0, 128, 128)
        self.y = self.x.copy()
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        self.dt = float(1/25)

    def initCond(self):
        self.vv = 1e3*np.exp(-((self.xx - 64) ** 2 + (self.yy - 64) ** 2)/100)
        self.vvold = self.vv.copy()

    def solve(self):

        u_list = []

        tc = 0
        dt = self.dt
        nstep = round(self.simulation_time / self.dt)+1
        r1, r2 = 15 ** 2 * (dt ** 2), 15 ** 2 * (dt ** 2)
        u_list.append(self.vvold)
        while tc < nstep:
            Z = self.vv.copy()
            xxx = np.linspace(self.x0, self.xf, self.grid_size)
            yyy = np.linspace(self.y0, self.yf, self.grid_size)
            self.xx, self.yy = np.meshgrid(xxx, yyy)
            vvnew = np.zeros((128,128))
            vvnew[0, :] = vvnew[-1, :] = vvnew[:, 0] = vvnew[:, -1] = 0

            # 在内部节点上使用五点差分法计算新的波场
            vvnew[1:-1, 1:-1] = 2 * self.vv[1:-1, 1:-1] - self.vvold[1:-1, 1:-1] + \
                            r1 * (self.vv[2:, 1:-1] - 2 * self.vv[1:-1, 1:-1] + self.vv[:-2, 1:-1]) + \
                            r2 * (self.vv[1:-1, 2:] - 2 * self.vv[1:-1, 1:-1] + self.vv[1:-1, :-2])
            self.vvold = self.vv.copy()
            self.vv = vvnew.copy()
            u_list.append(Z)
            tc=tc+1
        return np.asarray(u_list)


class NumericalSol:
    """
    Generating numerical solution for Schrödinger's 2D wave equation
    Parameters:
    --------------------------------------------------------------------------
    spatial_discretisation (int) : size of discrete resolution
    simulation_time (int) : number of seconds of simulation
    grid_size (int) : full size of sample grid
    Public variable:
    --------------------------------------------------------------------------
    dictionary of solution space:
    x, y, t, upper bound, lower bound, and u solution
    Returns:
    --------------------------------------------------------------------------
    numerical solution of u (array)
    """

    def __init__(self, simulation_time, grid_size):
        #self.spatial_discretisation = spatial_discretisation
        self.simulation_time = simulation_time
        self.grid_size = grid_size  # length of array

    def solve_numerical(self):
        simulator = WaveEquation(
            self.simulation_time, self.grid_size
        )
        self.u_sol = simulator.solve()


        dt = float(1/25) # spatial_discretisation and dt are fixed for ensuring numerical stability.

        lb = np.asarray([0.0, 0.0, 0])  # [x, y, t] Lower Bounds of the domain
        ub = np.asarray([128.0, 128.0, 64*float(float(1/25))])  # Upper Bounds of the domain

        x = np.linspace(0, 128, self.grid_size)
        y = x.copy()
        t = np.arange(lb[2], ub[2] + dt, dt)

        U_sol = self.u_sol

        # Storing the problem and solution information.
        self.sol_dict = {
            "x": x,
            "y": y,
            "t": t,
            "lower_range": lb,
            "upper_range": ub,
            "U_sol": U_sol,
        }

        return self.u_sol


# Fully Connected Network or a Multi-Layer Perceptron as the PINN.
class PINN(torch.nn.Module):
    """
    Creating neural network model
    Size:
    --------------------------------------------------------------------------
    3 inputs
    4 layers with 100 hidden nodes each
    1 output
    Activation always Tanh
    Returns:
    --------------------------------------------------------------------------
    Neural network model
    """



    def __init__(self, in_features, out_features, num_layers, num_neurons, activation=torch.tanh):
        super(PINN, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.num_neurons = num_neurons

        self.act_func = activation

        self.layers = torch.nn.ModuleList()

        self.layer_input = torch.nn.Linear(self.in_features, self.num_neurons)

        for ii in range(self.num_layers - 1):
            self.layers.append(torch.nn.Linear(self.num_neurons, self.num_neurons))
        self.layer_output = torch.nn.Linear(self.num_neurons, self.out_features)

    def forward(self, x):

        x_temp = self.act_func(self.layer_input(x)).requires_grad_(True)
        for dense in self.layers:
            x_temp = torch.tanh(dense(x_temp))
        x_temp = self.layer_output(x_temp)
        return x_temp

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c


class LossFunctions:
    """
    Calculation of loss functions for PINN
    Parameters:
    --------------------------------------------------------------------------
    simulation_time (int) : number of seconds of simulation
    model (torch.nn.Model) : NN representation
    Returns:
    --------------------------------------------------------------------------
    LHS_Sampling function -> Collocation point sampling (array)
    other functions -> return initial, boundary, pde, recostruction loss
    type is torch Tensor
    """

    def __init__(self, simulation_time, model):
        self.x_range = [1.0, 126.0]
        self.y_range = [1.0, 126.0]
        self.t_range = [0, 64*float(1/25)]
        self.D = 1.0

        self.model = model
        self.lb = np.asarray(
            [self.x_range[0], self.y_range[0], self.t_range[0]]
        )  # lower bounds
        self.ub = np.asarray(
            [self.x_range[1], self.y_range[1], self.t_range[1]]
        )  # Upper bounds

        # Setting up a derivative function that goes through the graph and calculates via chain rule the derivative of u wrt x
        self.derive = lambda u, x: \
        torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]

    def LHS_Sampling(self, sample_size):
        """
        Function to sample collocation points across the spatio-temporal domain
        using a Latin Hypercube
        """
        return self.lb + (self.ub - self.lb) * lhs(3, sample_size)

    def pde(self, X):
        """
        Domain Loss Function - measuring the deviation from the PDE functional.
        """
        x = X[:, 0:1]
        y = X[:, 1:2]
        t = X[:, 2:3]
        u = self.model(torch.cat([x, y, t], 1))

        u_x = self.derive(u, x)
        u_xx = self.derive(u_x, x)
        u_y = self.derive(u, y)
        u_yy = self.derive(u_y, y)
        u_t = self.derive(u, t)
        u_tt = self.derive(u_t, t)

        c_speed = 15.0
        #pde_loss = u_t ** 2 - (c_speed ** 2) * (u_x ** 2 +u_y ** 2)
        pde_loss = u_tt - (c_speed**2)*(u_xx + u_yy)
        #print("pde: ",pde_loss.pow(2).mean())

        return pde_loss.pow(2).mean()

    def boundary(self, X):
        """
        Boundary Loss Function - measuring the deviation from boundary conditions for f(x_lim, y_lim, t)
        """
        u = self.model(X)
        bc_loss = u - 0
        return bc_loss.pow(2).mean()

    def reconstruction(self, X, Y):
        """
        Reconstruction Loss Function - measuring the deviation fromt the actual output. Used to calculate the initial loss
        """
        u = self.model(X)
        recon_loss = u - Y
        return recon_loss.pow(2).mean()


class DataPrep:
    """
    Preparing data for training
    Parameters:
    --------------------------------------------------------------------------
    simulation_time (int) : number of seconds of simulation
    u_sol (array) : numerical solution of wave
    sol_dict (dict) : dictionary of numerical solution inputs and output
    sample_dict (dict) : dictionary of sample sizes, Ni, Nb, Nf
    model (tf.Keras.Model) : NN representation
    config_obj (class object) : torch configuration (default device)
    Returns:
    --------------------------------------------------------------------------
    data_list (list) : prepared sizes of inputs and outputs ready for training
    """

    def __init__(self, simulation_time, u_sol, sol_dict, sample_dict, model, config_obj):
        self.u_sol = u_sol
        self.x = sol_dict["x"]
        self.y = sol_dict["y"]
        self.t = sol_dict["t"]

        self.grid_length = len(self.x)

        # Samples taken from each region for optimisation purposes.
        self.N_i = sample_dict["N_i"]  # Initial
        self.N_b = sample_dict["N_b"]  # Boundary
        self.N_f = sample_dict["N_f"]  # Domain

        self.config_obj = config_obj

        self.loss_fnc = LossFunctions(simulation_time, model)

    def prep_io(self):
        self.u = np.asarray(self.u_sol)
        X, Y = np.meshgrid(self.x, self.y)
        self.XY_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

        T_star = np.expand_dims(np.repeat(self.t, len(self.XY_star)), 1)
        X_star_tiled = np.tile(self.XY_star, (len(self.t), 1))

        self.X_star = np.hstack((X_star_tiled, T_star))
        self.u_actual = np.expand_dims(self.u.flatten(), 1)

    def prep_initial(self):
        # X_IC = np.hstack(
        #     (self.XY_star, np.zeros(len(self.XY_star)).reshape(len(self.XY_star), 1))
        # )
        # u_IC = self.u[0].flatten()
        # u_IC = np.expand_dims(u_IC, 1)
        #
        # idx = np.random.choice(X_IC.shape[0], self.N_i, replace=False)
        # self.X_i = X_IC[idx]
        # self.u_i = u_IC[idx]

        self.X_i = torch.load("./x_i.pt").cuda()
        self.u_i = torch.load("./u_i.pt").cuda()

    def prep_boundary(self):
        # X_left = self.loss_fnc.LHS_Sampling(self.N_b)
        # X_left[:, 0:1] = self.loss_fnc.x_range[0]
        #
        # X_right = self.loss_fnc.LHS_Sampling(self.N_b)
        # X_right[:, 0:1] = self.loss_fnc.x_range[1]

        #
        # X_bottom = self.loss_fnc.LHS_Sampling(self.N_b)
        # X_bottom[:, 1:2] = self.loss_fnc.y_range[0]
        #
        # X_top = self.loss_fnc.LHS_Sampling(self.N_b)
        # X_top[:, 1:2] = self.loss_fnc.y_range[1]
        #
        # self.X_b = np.vstack((X_right, X_top, X_left, X_bottom))
        # np.random.shuffle(self.X_b)
        self.X_b = torch.load("./x_b.pt").cuda()

    def prep_domain(self):
        # self.X_f = torch.from_numpy(self.loss_fnc.LHS_Sampling(17232)).cuda()
        # self.X_f = torch.cat((torch.load("./x_f.pt").cuda(),self.X_f),dim=0).cpu().numpy()
        self.X_f = torch.load("./x_f.pt").cuda()
        # self.X_f = self.loss_fnc.LHS_Sampling(50000)
        # #print(self.X_f.shape)

    def convert_tensors(self):
        self.X_i = self.config_obj.torch_tensor_grad(self.X_i, self.config_obj.default_device)
        self.Y_i = self.config_obj.torch_tensor_nograd(self.u_i, self.config_obj.default_device)
        self.X_b = self.config_obj.torch_tensor_grad(self.X_b, self.config_obj.default_device)
        self.X_f = self.config_obj.torch_tensor_grad(self.X_f, self.config_obj.default_device)

    def prepare(self):
        self.prep_io()
        self.prep_initial()
        self.prep_boundary()
        self.prep_domain()
        self.convert_tensors()
        data_list = [self.X_star, self.u_actual, self.X_i, self.Y_i, self.X_b, self.X_f]
        return data_list


class Training:
    """
    Unsupervised training with customised training loss
    loss = initial loss + boundary loss + domain loss
    Parameters:
    --------------------------------------------------------------------------
    simulation_time (int) : number of seconds of simulation
    model (torch.nn.Model) : NN representation
    data_list (list) : list of array sizes for inputs and output
    epochs (int) : number of cyles of training
    config_obj (class object) : torch configuration (default device)
    Returns:
    --------------------------------------------------------------------------
    u_pred (array) : predicted output solution for PINN
    """

    def __init__(self, simulation_time, model, data_list, epochs, config_obj):
        self.model = model

        # random seed
        seed = 114514
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        #self.optimiser2 = torch.optim.LBFGS(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=5000, gamma=0.98)

        self.epochs = epochs
        self.config_obj = config_obj
        self.loss_list = []

        self.X_star, self.u_actual, self.X_i, self.Y_i, self.X_b, self.X_f = data_list

        self.loss_fnc = LossFunctions(simulation_time, self.model)

    def training_loop(self):
        it = 0
        start_time = time.time()

        while it < epochs:

            self.optimiser.zero_grad()
            initial_loss = self.loss_fnc.reconstruction(self.X_i, self.Y_i)
            boundary_loss = self.loss_fnc.boundary(self.X_b)
            domain_loss = self.loss_fnc.pde(self.X_f)

            loss = 1e3*(initial_loss + boundary_loss) + domain_loss
            self.loss_list.append(loss.item())

            accelerator.backward(loss)
            self.optimiser.step()
            self.scheduler.step()

            it += 1

            if it % 1000 == 0:
                with torch.no_grad():
                    u_pred = self.model(self.config_obj.torch_tensor_grad(self.X_star,
                                                                          self.config_obj.default_device)).cpu().detach().numpy()
                u_pred = u_pred.reshape(len(data.u), data.grid_length, data.grid_length)
                outputs1 = torch.unsqueeze(torch.from_numpy(u_pred), dim=1)
                torch.save(outputs1, './u_res2_'+str(it)+'.pt')

            print('Time: %.3f seconds, It: %d, Init: %.5f, Bound: %.3e, Domain: %.3e' % (
            time.time() - start_time, it, initial_loss.item(), boundary_loss.item(), domain_loss.item()))

        self.train_time = time.time() - start_time
        self.loss_list.append(loss)


        return self.loss_list

    def trained_output(self):
        if self.config_obj.default_device == 'cpu':
            with torch.no_grad():
                u_pred = self.model(
                    self.config_obj.torch_tensor_grad(self.X_star, self.config_obj.default_device)).detach().numpy()

        else:
            with torch.no_grad():
                u_pred = self.model(self.config_obj.torch_tensor_grad(self.X_star,
                                                                      self.config_obj.default_device)).cpu().detach().numpy()

        #l2_error = np.mean((self.u_actual - u_pred) ** 2)

        print("Training Time: %d seconds" % (self.train_time))
        u_pred = u_pred.reshape(len(data.u), data.grid_length, data.grid_length)
        outputs1 = torch.unsqueeze(torch.from_numpy(u_pred), dim=1)
        torch.save(outputs1, './u_res2_final.pt')

        return u_pred


class Plotting:
    """
    Generating plots
    1. L2 loss over epochs
    2. Numerical solution against PINN for 3 sample points
    Parameters:
    --------------------------------------------------------------------------
    lost_list (list) : generated list of all loss over epochs
    u_pred (array) : predicted output solution for PINN
    u_sol (array) : numerical solution of u (array)
    sol_dict (dict) : dictionary of numerical solution inputs and output
    """

    def training_loss(self, loss_list):
        plt.plot(loss_list)
        plt.xlabel("Iterations")
        plt.ylabel("L2 Error")
        plt.show()

    def num_vs_pinn(self, u_sol, u_pred, sol_dict):
        u_field = u_sol
        t = sol_dict["t"]

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(2, 3, 1)
        ax.imshow(u_field[0])
        ax.title.set_text("Initial")
        ax.set_ylabel("Solution")

        ax = fig.add_subplot(2, 3, 2)
        ax.imshow(u_field[int(len(t) / 2)])
        ax.title.set_text("Middle")

        ax = fig.add_subplot(2, 3, 3)
        ax.imshow(u_field[-1])
        ax.title.set_text("Final")

        u_field = u_pred

        ax = fig.add_subplot(2, 3, 4)
        ax.imshow(u_field[0])
        ax.set_ylabel("PINN")

        ax = fig.add_subplot(2, 3, 5)
        ax.imshow(u_field[int(len(t) / 2)])

        ax = fig.add_subplot(2, 3, 6)
        ax.imshow(u_field[-1])

        plt.show()


if __name__ == "__main__":
    config = TorchConfig()

    command_line = CommandLineArgs()

    epochs = 50000
    #spartial_discretisation = command_line.args.n_steps
    simulation_time = command_line.args.time
    grid_size = command_line.args.grid_size
    sample_dict = {
        "N_i": command_line.args.initial,
        "N_b": command_line.args.boundary,
        "N_f": command_line.args.domain,
    }

    numerical_sol = NumericalSol(simulation_time, grid_size)
    u_sol = numerical_sol.solve_numerical()
    outputs1 = torch.unsqueeze(torch.from_numpy(u_sol), dim=1)
    print(outputs1.shape)
    u_sol = u_sol[1:]
    torch.save(outputs1, './u_sol_'+str(epochs)+'.pt')

    model = PINN(in_features=3, out_features=1, num_layers=5, num_neurons=64)
    model = model.to(config.default_device)

    data = DataPrep(simulation_time, u_sol, numerical_sol.sol_dict, sample_dict, model, config)
    data_list = data.prepare()

    train_model = Training(simulation_time, model, data_list, epochs, config)
    lost_list = train_model.training_loop()

    u_pred = train_model.trained_output()

    # plots = Plotting()
    #plots.training_loss(lost_list)
    # plots.num_vs_pinn(u_sol, u_pred, numerical_sol.sol_dict)