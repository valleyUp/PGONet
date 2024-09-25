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
import ffn
from fno import FNO2d
from cno import CNO2d, UNet2d
from kno import KNO2d

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

class LossFunctions:
    def __init__(self, c, dt, dx, dy, t_max, init):
        self.c = c
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.t_max = t_max
        self.init = torch.cat((init,init),dim=0)
    def pde(self, output7):
        """
        Domain Loss Function - measuring the deviation from the PDE functional.
        """
        r1, r2 = self.c ** 2 * self.dt ** 2 / self.dx ** 2, self.c ** 2 * self.dt ** 2 / self.dy ** 2
        output1 = output7.clone()
        temp = torch.zeros_like(output1[:,:,1:-1,1:-1]).cuda()
        temp1 = torch.zeros_like(output1[:, :, 1:-1, 1:-1]).cuda()
        output1 = torch.cat((self.init,output1),dim=0)
        n = int(self.t_max/self.dt)
        temp[:,:, :, :] = -output7[:,:,1:-1,1:-1] + 2 * output1[1:n+1, :,1:-1, 1:-1] - output1[0:n, :,1:-1, 1:-1] + \
                                           r1 * (
                                                       output1[1:n+1, :,2:, 1:-1] - 2 * output1[1:n+1, :,1:-1, 1:-1] + output1[1:n+1,:,
                                                                                                           :-2, 1:-1]) + \
                                           r2 * (
                                                       output1[1:n+1, :,1:-1, 2:] - 2 * output1[1:n+1,:, 1:-1, 1:-1] + output1[1:n+1,:,
                                                                                                           1:-1, :-2])

        pde_loss = torch.nn.functional.mse_loss(temp[:],temp1[:])
        init_loss = 0
        return pde_loss, init_loss

    def boundary(self, output1):
        """
        Boundary Loss Function - measuring the deviation from boundary conditions for f(x_lim, y_lim, t)
        """
        bc_loss = torch.sum(output1[:,:,0,:]**2)+torch.sum(output1[:,:,-1,:]**2)+torch.sum(output1[:,:,:,0]**2)+torch.sum(output1[:,:,:,-1]**2)

        return bc_loss/(4*self.t_max*128)

torch.autograd.set_detect_anomaly(True)
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
    outputs1 = torch.unsqueeze(torch.from_numpy(u_sol), dim=0).float().cuda()
    print(outputs1.shape)
    #torch.save(outputs1, './u_sol_'+str(epochs)+'.pt')
    init = outputs1[0:1,0:1,:,:].clone()

    loss_fnc = LossFunctions(15, float(1/25), 1, 1, 64*float(1/25), init)

    # model = UNet2d(in_channel=1,
    #             out_channel=64,
    #             hidden_channel=64,
    #             num_layers=5,
    #             kernel_size=3,
    #             depth=3,
    #             activation="relu")
    # model = KNO2d(in_channel=1,
    #                out_channel=64,
    #                hidden_channel=64,
    #                num_layers=5,
    #                )
    # model = FNO2d(in_channel=1,
    #              out_channel=64,
    #              hidden_channel=64,
    #              num_layers=5,
    #              activation='gelu',
    #              modes=None,
    #              io_transform=False)
    model = CNO2d(in_channel=1,
                  out_channel=64,
                  hidden_channel=64,
                  num_layers=5,
                  kernel_size=3,
                  depth=3,
                  jit=True,
                  activation="relu")
    model = model.to(config.default_device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # self.optimiser2 = torch.optim.LBFGS(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=5000, gamma=0.98)

    start_time = time.time()
    for i in range(epochs):
        optimiser.zero_grad()
        outputs1 = model(init.cuda())
        outputs1 = torch.squeeze(outputs1, dim=0).cuda()
        outputs1 = torch.unsqueeze(outputs1, dim=1).cuda()
        boundary_loss = loss_fnc.boundary(outputs1.cuda())
        domain_loss,init_loss = loss_fnc.pde(outputs1.cuda())

        loss = domain_loss

        accelerator.backward(loss)
        optimiser.step()
        scheduler.step()
        print('Time: %.3f seconds, It: %d, Loss:%.3e, Bound: %.3e, Init: %.3e, Domain: %.3e' % (
            time.time() - start_time, i, loss.item(), boundary_loss.item(), 0, domain_loss.item()))

        if (i+1)%10000 == 0:
            outputs1 = torch.cat((init,outputs1),dim=0)
            print(outputs1.shape)
            torch.save(outputs1, './u_res2_'+str(i+1)+'_CNO.pt')