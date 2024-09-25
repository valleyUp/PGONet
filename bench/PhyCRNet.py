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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os
from torch.nn.utils import weight_norm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
__author__ = "Lucy Harris, Vignesh Gopakumar"
__license__ = "GPL 2"
__email__ = "lucy.harris@ukaea.uk"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)

# define the high-order finite difference kernels
lapl_op = [[[[0, 0, -1 / 12, 0, 0],
             [0, 0, 4 / 3, 0, 0],
             [-1 / 12, 4 / 3, -5, 4 / 3, -1 / 12],
             [0, 0, 4 / 3, 0, 0],
             [0, 0, -1 / 12, 0, 0]]]]

partial_y = [[[[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [1 / 12, -8 / 12, 0, 8 / 12, -1 / 12],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]]]

partial_x = [[[[0, 0, 1 / 12, 0, 0],
               [0, 0, -8 / 12, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 8 / 12, 0, 0],
               [0, 0, -1 / 12, 0, 0]]]]


# generalized version
# def initialize_weights(module):
#     ''' starting from small initialized parameters '''
#     if isinstance(module, nn.Conv2d):
#         c = 0.1
#         module.weight.data.uniform_(-c*np.sqrt(1 / np.prod(module.weight.shape[:-1])),
#                                      c*np.sqrt(1 / np.prod(module.weight.shape[:-1])))

#     elif isinstance(module, nn.Linear):
#         module.bias.data.zero_()

# specific parameters for burgers equation
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        # nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
        c = 1  # 0.5
        module.weight.data.uniform_(-c * np.sqrt(1 / (3 * 3 * 320)),
                                    c * np.sqrt(1 / (3 * 3 * 320)))

    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class ConvLSTMCell(nn.Module):
    ''' Convolutional LSTM '''

    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                 input_stride, input_padding):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_kernel_size = 3
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.num_features = 4

        # padding for hidden state
        self.padding = int((self.hidden_kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels,
                             self.input_kernel_size, self.input_stride, self.input_padding,
                             bias=True, padding_mode='circular')

        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                             self.hidden_kernel_size, 1, padding=1, bias=False,
                             padding_mode='circular')

        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels,
                             self.input_kernel_size, self.input_stride, self.input_padding,
                             bias=True, padding_mode='circular')

        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                             self.hidden_kernel_size, 1, padding=1, bias=False,
                             padding_mode='circular')

        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels,
                             self.input_kernel_size, self.input_stride, self.input_padding,
                             bias=True, padding_mode='circular')

        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                             self.hidden_kernel_size, 1, padding=1, bias=False,
                             padding_mode='circular')

        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels,
                             self.input_kernel_size, self.input_stride, self.input_padding,
                             bias=True, padding_mode='circular')

        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                             self.hidden_kernel_size, 1, padding=1, bias=False,
                             padding_mode='circular')

        nn.init.zeros_(self.Wxi.bias)
        nn.init.zeros_(self.Wxf.bias)
        nn.init.zeros_(self.Wxc.bias)
        self.Wxo.bias.data.fill_(1.0)

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)

        return ch, cc

    def init_hidden_tensor(self, prev_state):
        return (Variable(prev_state[0]).cuda(), Variable(prev_state[1]).cuda())


class encoder_block(nn.Module):
    ''' encoder with CNN '''

    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                 input_stride, input_padding):
        super(encoder_block, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding

        self.conv = weight_norm(nn.Conv2d(self.input_channels,
                                          self.hidden_channels, self.input_kernel_size, self.input_stride,
                                          self.input_padding, bias=True, padding_mode='circular'))

        self.act = nn.ReLU()

        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.act(self.conv(x))


class PhyCRNet(nn.Module):
    ''' physics-informed convolutional-recurrent neural networks '''

    def __init__(self, input_channels, hidden_channels,
                 input_kernel_size, input_stride, input_padding, dt,
                 num_layers, upscale_factor, step=1, effective_step=[1]):

        super(PhyCRNet, self).__init__()

        # input channels of layer includes input_channels and hidden_channels of cells
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.dt = dt
        self.upscale_factor = upscale_factor

        # number of layers
        self.num_encoder = num_layers[0]
        self.num_convlstm = num_layers[1]

        # encoder - downsampling
        for i in range(self.num_encoder):
            name = 'encoder{}'.format(i)
            cell = encoder_block(
                input_channels=self.input_channels[i],
                hidden_channels=self.hidden_channels[i],
                input_kernel_size=self.input_kernel_size[i],
                input_stride=self.input_stride[i],
                input_padding=self.input_padding[i])

            setattr(self, name, cell)
            self._all_layers.append(cell)

            # ConvLSTM
        for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
            name = 'convlstm{}'.format(i)
            cell = ConvLSTMCell(
                input_channels=self.input_channels[i],
                hidden_channels=self.hidden_channels[i],
                input_kernel_size=self.input_kernel_size[i],
                input_stride=self.input_stride[i],
                input_padding=self.input_padding[i])

            setattr(self, name, cell)
            self._all_layers.append(cell)

            # output layer
        self.output_layer = nn.Conv2d(1, 1, kernel_size=5, stride=1,
                                      padding=2, padding_mode='circular')

        # pixelshuffle - upscale
        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)

        # initialize weights
        self.apply(initialize_weights)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, initial_state, x):

        self.initial_state = initial_state
        internal_state = []
        outputs = []
        second_last_state = []
        xt = x
        for step in range(self.step):
            x_tt = xt
            xt = x

            # encoder
            for i in range(self.num_encoder):
                name = 'encoder{}'.format(i)
                x = getattr(self, name)(x)

            # convlstm
            for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
                name = 'convlstm{}'.format(i)
                if step == 0:
                    (h, c) = getattr(self, name).init_hidden_tensor(
                        prev_state=self.initial_state[i - self.num_encoder])
                    internal_state.append((h, c))

                # one-step forward
                (h, c) = internal_state[i - self.num_encoder]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i - self.num_encoder] = (x, new_c)

                # output
            x = self.pixelshuffle(x)
            x = self.output_layer(x)

            r = 15 ** 2 * (float(1/25) ** 2)
            # residual connection
            x = 2 * xt - x_tt + r * x

            x[:, :, -1, :] = 0
            x[:, :, 0, :] = 0
            x[:, :, :, -1] = 0
            x[:, :, :, 0] = 0
            if step == (self.step - 2):
                second_last_state = internal_state.copy()

            if step in self.effective_step:
                outputs.append(x)

        return outputs, second_last_state


class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol

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

        pde_loss = torch.nn.functional.mse_loss(temp[:],temp1[:])/(self.dt ** 2)
        #init_loss = torch.nn.functional.mse_loss(temp[0:1],temp1[0:1])
        return pde_loss, 0

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
    init = outputs1[0:1,0:1,:,:]

    loss_fnc = LossFunctions(15, float(1/25), 1, 1, 64*float(1/25), init)

    # model = UNet2d(in_channel=1,
    #             out_channel=64,
    #             hidden_channel=64,
    #             num_layers=5,
    #             kernel_size=3,
    #             depth=3,
    #             activation="relu")
    # model = FNO2d(in_channel=1,
    #              out_channel=64,
    #              hidden_channel=64,
    #              num_layers=5,
    #              activation='gelu',
    #              modes=None,
    #              io_transform=False)
    time_steps = 64
    steps = time_steps
    effective_step = list(range(0, steps))
    model = PhyCRNet(
        input_channels = 1,
        hidden_channels = [4, 32, 64, 64],
        input_kernel_size = [4, 4, 4, 3],
        input_stride = [2, 2, 2, 1],
        input_padding = [1, 1, 1, 1],
        dt = float(1/25),
        num_layers = [3, 1],
        upscale_factor = 8,
        step = steps,
        effective_step = effective_step).cuda()

    model = model.to(config.default_device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # self.optimiser2 = torch.optim.LBFGS(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=5000, gamma=0.98)
    num_convlstm = 1
    (h0, c0) = (torch.randn(1, 64, 16, 16), torch.randn(1, 64, 16, 16))
    initial_state = []
    for i in range(num_convlstm):
        initial_state.append((h0, c0))

    start_time = time.time()

    for i in range(epochs):
        hidden_state = initial_state
        optimiser.zero_grad()
        outputs1,_ = model(initial_state,init.cuda())
        outputs1 = torch.cat(tuple(outputs1), dim=0)
        # outputs1 = torch.squeeze(outputs1, dim=0).cuda()
        # outputs1 = torch.unsqueeze(outputs1, dim=1).cuda()
        boundary_loss = loss_fnc.boundary(outputs1.cuda())
        domain_loss,init_loss = loss_fnc.pde(outputs1.cuda())

        loss = 1e3 * (boundary_loss + init_loss) + domain_loss

        accelerator.backward(loss)
        optimiser.step()
        scheduler.step()
        print('Time: %.3f seconds, It: %d, Loss:%.3e, Bound: %.3e, Init: %.3e, Domain: %.3e' % (
            time.time() - start_time, i, loss.item(), boundary_loss.item(), 0, domain_loss.item()))

        if (i+1)%5000 == 0:
            outputs1 = torch.cat((init,outputs1),dim=0)
            print(outputs1.shape)
            torch.save(outputs1, './u_res2_'+str(i+1)+'_PhyCRNet.pt')