

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.io as scio
import time
import datetime
import os
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from matplotlib import cm
from accelerate import Accelerator

accelerator = Accelerator()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)

# define the high-order finite difference kernels

lapl_op2 = [[[[    0,   1, 0],
             [    1,   -4,   1],
             [0, 1,    0]]]]

avg_op1 = torch.tensor([[[
             [    1/4,   1/2,   1/4]]]]).to(accelerator.device)
avg_op11 = torch.tensor([[[
             [    1/16, 3/16,   1/2,   3/16, 1/16]]]]).to(accelerator.device)
avg_op2 = torch.tensor([[[[    1/16,   1/16, 1/16],
             [    1/16,   1/2,   1/16],
             [1/16, 1/16,  1/16]]]]).to(accelerator.device)
avg_op21 = torch.tensor([[[
            [ 1/64,  1/64,   1/64, 1/64, 1/64],
            [ 1/64,  1/32,   1/32, 1/32, 1/64],
             [1/64, 1/32,   1/2,   1/32, 1/64],
             [1/64, 1/32, 1/32,  1/32, 1/64],
    [1 / 64, 1 / 64, 1 / 64, 1 / 64, 1 / 64]
]]]).to(accelerator.device)
solve = []
class SteepSigmoid(nn.Module):
    def __init__(self, beta=10):
        super(SteepSigmoid, self).__init__()
        self.beta = beta

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.beta * x))

# specific parameters for burgers equation
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        # nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
        c = 1  # 0.5
        module.weight.data.uniform_(-c * np.sqrt(1 / (3 * 3 * 320)),
                                    c * np.sqrt(1 / (3 * 3 * 320)))

    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

# 定义Laplace算子的权重
laplace_kernel = torch.tensor([[[[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]]]], dtype=torch.float32)

# 创建卷积层，但权重和偏置设为不可训练
class LaplaceConv2d(nn.Module):
    def __init__(self):
        super(LaplaceConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
        self.conv.weight.data = laplace_kernel
        self.conv.weight.requires_grad = False  # 设置为不可训练

    def forward(self, x):
        return self.conv(x)

class PreNet(nn.Module):
    def __init__(self, n, sigmoid_n):

        super(PreNet, self).__init__()
        self.num = n
        self.sigmoid_n = sigmoid_n
        self.ref_sol = torch.load('./case/SeaFloor1/o_temp.pt').to(accelerator.device)
        self.ref_sol2 = torch.zeros((self.ref_sol.shape[0], 1, 3, 9)).to(accelerator.device)

        self.fc74 = nn.Linear(3 * self.ref_sol2.shape[2] * self.ref_sol2.shape[3], 64)  # 第二层全连接层
        self.fc75 = nn.Linear(64, 64)  # 第二层全连接层
        self.fc76 = nn.Linear(64, 64*self.num)
        self.index_matrix = self.generate_index_matrix(64*n-2).to(accelerator.device).detach()
        self.steep_sigmoid = SteepSigmoid(beta=1)
        self.fix_layer = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, padding_mode='circular')
        self.fix_layer.weight.requires_grad = False
        self.fix_layer.bias.requires_grad = False
        self.fix_layer.weight.data = avg_op21
        self.fix_layer2 = nn.Conv2d(1, 1, kernel_size=(1,3), stride=1, padding_mode='zeros')
        self.fix_layer2.weight.requires_grad = False
        self.fix_layer2.bias.requires_grad = False
        self.fix_layer2.weight.data = avg_op1
        self.fix_layer21 = nn.Conv2d(1, 1, kernel_size=(1, 5), stride=1, padding_mode='zeros')
        self.fix_layer21.weight.requires_grad = False
        self.fix_layer21.bias.requires_grad = False
        self.fix_layer21.weight.data = avg_op11

    def generate_index_matrix(self, n):
        # 创建一个从0到n-1的一维张量
        row_indices = torch.arange(n)

        index_matrix = row_indices.unsqueeze(1).repeat(1, n)

        return index_matrix

    def get_local_Loss(self, res):
        res1 = res.clone()
        local_res = torch.zeros((res.shape[0], 1, 3, 9)).to(accelerator.device)
        local_res[:, :, 0, 0] = res1[:, :, int(40 * self.num), 2 * self.num]
        local_res[:, :, 0, 1] = res1[:, :, int(40 * self.num), 5 * self.num]
        local_res[:, :, 0, 2] = res1[:, :, int(40 * self.num), 20 * self.num]
        local_res[:, :, 0, 3] = res1[:, :, int(40 * self.num), 24 * self.num]
        local_res[:, :, 0, 4] = res1[:, :, int(40 * self.num), 32 * self.num]
        local_res[:, :, 0, 5] = res1[:, :, int(40 * self.num), 40 * self.num]
        local_res[:, :, 0, 6] = res1[:, :, int(40 * self.num), 48 * self.num]
        local_res[:, :, 0, 7] = res1[:, :, int(40 * self.num), 52 * self.num]
        local_res[:, :, 0, 8] = res1[:, :, int(40 * self.num), 55 * self.num]

        local_res[:, :, 2, 0] = res1[:, :, int(35 * self.num), 2 * self.num]
        local_res[:, :, 2, 1] = res1[:, :, int(35 * self.num), 8 * self.num]
        local_res[:, :, 2, 2] = res1[:, :, int(35 * self.num), 16 * self.num]
        local_res[:, :, 2, 3] = res1[:, :, int(35 * self.num), 24 * self.num]
        local_res[:, :, 2, 4] = res1[:, :, int(35 * self.num), 32 * self.num]
        local_res[:, :, 2, 5] = res1[:, :, int(35 * self.num), 40 * self.num]
        local_res[:, :, 2, 6] = res1[:, :, int(35 * self.num), 48 * self.num]
        local_res[:, :, 2, 7] = res1[:, :, int(35 * self.num), 56 * self.num]
        local_res[:, :, 2, 8] = res1[:, :, int(35 * self.num), 62 * self.num]

        local_res[:, :, 1, 0] = res1[:, :, 45 * self.num, 1 * self.num]
        local_res[:, :, 1, 1] = res1[:, :, 45 * self.num, 3 * self.num]
        local_res[:, :, 1, 2] = res1[:, :, 45 * self.num, 22 * self.num]
        local_res[:, :, 1, 3] = res1[:, :, 45 * self.num, 25 * self.num]
        local_res[:, :, 1, 4] = res1[:, :, 45 * self.num, 29 * self.num]
        local_res[:, :, 1, 5] = res1[:, :, 45 * self.num, 32 * self.num]
        local_res[:, :, 1, 6] = res1[:, :, 45 * self.num, 35 * self.num]
        local_res[:, :, 1, 7] = res1[:, :, 45 * self.num, 58 * self.num]
        local_res[:, :, 1, 8] = res1[:, :, 45 * self.num, 62 * self.num]

        return local_res

    def forward(self, output, bsize, ntb):
        ref_speed = 1500 * torch.ones((1, 1, 64*self.num, 64*self.num)).to(accelerator.device)
        ref_2 = torch.zeros((1, 1, 64*self.num)).to(accelerator.device)

        t_ref_sol = self.get_local_Loss(output)
        for idx in range(ntb):
            for i in range(bsize):
                temp = torch.reshape(t_ref_sol[bsize*idx+i:bsize*idx+i+3, :, :, :], (1, 3 * self.ref_sol2.shape[2] * self.ref_sol2.shape[3]))
                c5 = torch.tanh(self.fc74(temp))
                c5 = torch.tanh(self.fc75(c5))
                ref_2 = (ref_2 * (i + idx * bsize) + torch.reshape(30 * self.num + 34 * self.num * torch.sigmoid(self.fc76(c5)), (1, 1, 64*self.num))) \
                        / (i + idx * bsize + 1)

        ref_2 = self.fix_layer2(ref_2)
        ref_2[:,:,2:-2] = self.fix_layer21(ref_2.clone())
        ref_2 = ref_2.squeeze(dim=1).repeat(64*self.num-2, 1)
        t_ref_speed = 1500 - 1500 * (1 / (1 + torch.exp(self.sigmoid_n * (self.index_matrix - ref_2)))).unsqueeze(dim=0).unsqueeze(dim=0)
        t_ref_speed = self.fix_layer(t_ref_speed)
        ref_speed[:, :, 1:-1, 1:-1] = t_ref_speed

        return ref_speed

class PGONet(nn.Module):
    ''' physics-informed convolutional-recurrent neural networks '''

    def __init__(self, dt, dx, fre):

        super(PGONet, self).__init__()
        #self.laplace_conv = LaplaceConv2d()
        self.fre = fre
        # input channels of layer includes input_channels and hidden_channels of cells
        self.backward_state = None
        self.forward_state = None
        self.dt = dt
        self.dx = dx

        # ConvLSTM(Forward)
        self.input_layer9 = weight_norm(nn.Conv2d(3, 1, kernel_size=(3,3), stride=1,
                                                padding=1))


        self.ref_sol = torch.load('./case/SeaFloor1/o_temp.pt').to(accelerator.device)
        self.test_ref_speed = torch.load('./case/SeaFloor1/ref_speed.pt').unsqueeze(dim=0).unsqueeze(dim=0).to(accelerator.device)
        self.apply(initialize_weights)
        self.internal_state_forward = []

    def forward(self, ref_speed, bsize, id, loc_x, loc_y, batch,
                flag, flag_num, x_tt, x_t):
        outputs1 = []
        outputs2 = []
        outputs3 = []
        outputs4 = []
        # ref_speed = self.test_ref_speed
        outputs1.append(x_tt)
        outputs1.append(x_t)
        outputs2.append(x_tt)
        outputs2.append(x_t)

        ntb = flag - 1
        step = flag_num - 1
        # x_tt = batch[ntb * bsize + step:ntb * bsize + step + 1].detach()
        # x_t = batch[ntb * bsize + step + 1:ntb * bsize + step + 2].detach()

        x_t4 = torch.zeros_like(x_t).to(accelerator.device)
        x_t4[:, :, 1:-1, 1:-1] = ((2 * x_t[:, :, 1:-1, 1:-1] - x_tt[:, :, 1:-1, 1:-1]) +
                                  (x_t[:, :, 2:, 1:-1] - 4 * x_t[:, :, 1:-1, 1:-1] + x_t[:, :, :-2,
                                                                                     1:-1]
                                   + x_t[:, :, 1:-1, 2:] + x_t[:, :, 1:-1, :-2]) * (
                                          ref_speed[:, :, 1:-1, 1:-1].detach() ** 2) * (self.dt ** 2) / (self.dx ** 2))

        outputs3.append(x_t4)

        x_t1 = torch.concat((x_tt, x_t, ref_speed), dim=1)

        x_temp7 = self.input_layer9(x_t1)
        x_temp7[:, :, 1:-1, 1:-1] = ((2 * x_t[:, :, 1:-1, 1:-1] - x_tt[:, :, 1:-1, 1:-1]) +
                                     (x_temp7[:, :, 2:, 1:-1] - 4 * x_temp7[:, :, 1:-1, 1:-1] + x_temp7[:, :, :-2,
                                                                                                1:-1] + x_temp7[:,
                                                                                                        :, 1:-1,
                                                                                                        2:] + x_temp7[
                                                                                                              :, :,
                                                                                                              1:-1,
                                                                                                              :-2]) * (
                                             ref_speed[:, :, 1:-1, 1:-1] ** 2) * (self.dt ** 2) / (
                                             self.dx ** 2))
        outputs2.append(x_temp7.clone())
        x_temp7[:, :, 0, :] = 0
        x_temp7[:, :, :, 0:1] = x_t[:, :, :, 0:1] - self.dt * ref_speed[:, :, :, 0:1] * (
                x_t[:, :, :, 0:1] - x_t[:, :, :, 1:2]) / self.dx

        x_temp7[:, :, :, -1:] = x_t[:, :, :, -1:] - self.dt * ref_speed[:, :, :, -1:] * \
                                (x_t[:, :, :, -1:] - x_t[:, :, :, -2:-1]) / self.dx

        for idx in range(len(loc_x)):
            if loc_x[idx] != -1:
                x_temp7[:, :, int(loc_x[idx]), int(loc_y[idx])] = self.ref_sol[
                                                                  id * bsize + step + 2:id * bsize + step + 3,
                                                                  :, int(loc_x[idx]), int(
                    loc_y[idx])]  # 1500 * np.sin(2 * 3.1415926 * self.fre * (bsize * id + step + 2) * dt)

        outputs1.append(x_temp7.clone())

        second_last_state_forward = self.internal_state_forward.copy()

        outputs1 = torch.cat(tuple(outputs1), dim=0)
        outputs2 = torch.cat(tuple(outputs2), dim=0)
        outputs3 = torch.cat(tuple(outputs3), dim=0)

        return outputs4, outputs1, outputs2, outputs3, second_last_state_forward, self.test_ref_speed


class loss_generator(nn.Module):
    ''' Loss generator for physics loss '''

    def __init__(self, num, dtt, dxx, fre):
        ''' Construct the derivatives, X = Width, Y = Height '''

        super(loss_generator, self).__init__()

        # spatial derivative operator
        self.flag =False

        self.ref_sol = torch.load('./case/SeaFloor1/o_temp.pt').to(accelerator.device)
        self.num = num
        self.dttt = dtt
        self.dxx = dxx
        self.fre = fre


    def get_ref_Loss(self):
        temp_res = self.ref_sol[:,:,:,:]
        return temp_res

    def get_local_Loss(self, res):
        res1 = res.clone()
        local_res = torch.zeros((res.shape[0], 1, 3, 9)).to(accelerator.device)
        local_res[:, :, 0, 0] = res1[:, :, int(40 * self.num), 2 * self.num]
        local_res[:, :, 0, 1] = res1[:, :, int(40 * self.num), 5 * self.num]
        local_res[:, :, 0, 2] = res1[:, :, int(40 * self.num), 20 * self.num]
        local_res[:, :, 0, 3] = res1[:, :, int(40 * self.num), 24 * self.num]
        local_res[:, :, 0, 4] = res1[:, :, int(40 * self.num), 32 * self.num]
        local_res[:, :, 0, 5] = res1[:, :, int(40 * self.num), 40 * self.num]
        local_res[:, :, 0, 6] = res1[:, :, int(40 * self.num), 48 * self.num]
        local_res[:, :, 0, 7] = res1[:, :, int(40 * self.num), 52 * self.num]
        local_res[:, :, 0, 8] = res1[:, :, int(40 * self.num), 55 * self.num]

        local_res[:, :, 2, 0] = res1[:, :, int(35 * self.num), 2 * self.num]
        local_res[:, :, 2, 1] = res1[:, :, int(35 * self.num), 8 * self.num]
        local_res[:, :, 2, 2] = res1[:, :, int(35 * self.num), 16 * self.num]
        local_res[:, :, 2, 3] = res1[:, :, int(35 * self.num), 24 * self.num]
        local_res[:, :, 2, 4] = res1[:, :, int(35 * self.num), 32 * self.num]
        local_res[:, :, 2, 5] = res1[:, :, int(35 * self.num), 40 * self.num]
        local_res[:, :, 2, 6] = res1[:, :, int(35 * self.num), 48 * self.num]
        local_res[:, :, 2, 7] = res1[:, :, int(35 * self.num), 56 * self.num]
        local_res[:, :, 2, 8] = res1[:, :, int(35 * self.num), 62 * self.num]

        local_res[:, :, 1, 0] = res1[:, :, 45 * self.num, 1 * self.num]
        local_res[:, :, 1, 1] = res1[:, :, 45 * self.num, 3 * self.num]
        local_res[:, :, 1, 2] = res1[:, :, 45 * self.num, 22 * self.num]
        local_res[:, :, 1, 3] = res1[:, :, 45 * self.num, 25 * self.num]
        local_res[:, :, 1, 4] = res1[:, :, 45 * self.num, 29 * self.num]
        local_res[:, :, 1, 5] = res1[:, :, 45 * self.num, 32 * self.num]
        local_res[:, :, 1, 6] = res1[:, :, 45 * self.num, 35 * self.num]
        local_res[:, :, 1, 7] = res1[:, :, 45 * self.num, 58 * self.num]
        local_res[:, :, 1, 8] = res1[:, :, 45 * self.num, 62 * self.num]

        return local_res

    def get_phy_Loss1(self, model, output, c, bsize1, id2, loc_x, loc_y, coffe):
        output3 = torch.zeros_like(output[:2+int(coffe*bsize1), :, :, :]).to(accelerator.device)
        output3[0:1, :,:, :] = output[0:1, :, :, :]
        output3[1:2, :,:, :] = output[1:2, :, :, :]
        for flag_num in range(0, output3.shape[0]-2, 1):
            _, output1, _, _, _, _ \
                = model(c, int(bsize1*coffe), id2, loc_x, loc_y,
                        output3, 1, flag_num+1, output3[flag_num:flag_num+1], output3[flag_num+1:flag_num+2])
            #print(F.mse_loss(output1[2:3],output[2+flag_num:3+flag_num]))
            output3[2+flag_num:3+flag_num] = output1[2:3].clone()
            #print(F.mse_loss(output1[2:3], output3[2 + flag_num:3 + flag_num]))
        return output3[2:, :, :, :]

def compute_loss(output71, output2, output3, loss_func, id, id2,
                 bsize, bsize1, coffe, flag_num,
                 batch,history_loss, t_epoch,num_batch_size2,last_loss_weight):
    ''' calculate the phycis loss '''

    mse_loss = nn.MSELoss(reduction='mean')

    x_tt = output71[-2:-1].clone().detach()
    x_t = output71[-1:].clone().detach()
    t_flag = False
    i = flag_num - 1
    p_local2 = mse_loss(
        loss_func.ref_sol[id * (bsize + 2) + id2 * bsize1 + 2+i:id * (bsize + 2) + id2 * bsize1 + 3+i, :, :, :],
        output71[2:3, :, :, :])
    ref_local = loss_func.get_local_Loss(loss_func.ref_sol[id * (bsize + 2) + id2 * bsize1 + 2+i:id * (bsize + 2) + id2 * bsize1 + 3+i, :, :, :])
    p_res = mse_loss(output2[2:3,:,1:-1,1:-1],output3[0:1,:,1:-1,1:-1].detach())
    p_local = mse_loss(ref_local,loss_func.get_local_Loss(output71[2:3, :, :, :]))

    if p_res < history_loss[id * (bsize + 2) + id2 * bsize1 + 2 + i]:
        history_loss[id * (bsize + 2) + id2 * bsize1 + 2 + i] = p_res
        if id2 >= coffe * num_batch_size2:
            batch[id * (bsize + 2) + id2 * bsize1 + 2 + i:id * (bsize + 2) + id2 * bsize1 + 3 + i] =output71[2: 3]

    if i==flag_num-1 and (((t_epoch>=300 or p_res< 1) and (id2!=0 or i!=0)) or t_epoch == 3000):
        flag_num += 1
        t_flag =True
        t_epoch = 0

    t_loss = 1 / (max(last_loss_weight, 1)) * p_local + p_res
    return t_loss, p_local, p_res, p_local2, flag_num, batch, t_flag, t_epoch, x_tt, x_t

def compute_loss_p(model, output71, loss_func, ntb, ref_speed, bsize, last_loss_weight, size_batch, last_ref_speed, coffe, epoch):
    ''' calculate the phycis loss '''

    mse_loss = nn.MSELoss(reduction='mean')
    x1 = np.load('./case/SeaFloor1/x1.npy')
    y1 = np.load('./case/SeaFloor1/y1.npy')
    ref_local_sol = loss_func.get_ref_Loss().to(accelerator.device)
    output72 = output71.clone()
    loss = 0
    local_ref_speed = loss_func.get_local_Loss(ref_speed)
    p_speed = mse_loss(ref_speed, last_ref_speed) #+ mse_loss(local_ref_speed, 1500*torch.ones_like(local_ref_speed))
    num_time_batch2 = int((bsize) / size_batch)
    output_t = None
    x_tt1 = None
    x_t1 = None
    p_local = 0
    p_sim = 0
    #ref_local_speed = loss_func.get_local_Loss(ref_speed)
    for id in range(ntb):
        for idx in range(num_time_batch2):
            output7 = output72[id*(bsize+2)+idx*size_batch:id*(bsize+2)+(idx+1)*size_batch+2].clone()
            output714 = output71[id*(bsize+2)+idx*size_batch:id*(bsize+2)+(idx+1)*size_batch+2].clone()
            if idx!=0:
                output714[0:1] = x_tt1
                output714[1:2] = x_t1
            output11_3 = loss_func.get_phy_Loss1(model, output714.clone(), ref_speed, size_batch, idx, x1[id], y1[id], 1)
            x_tt1 = output11_3[-2:-1].clone()
            x_t1 = output11_3[-1:].clone()
            output81 = output71[id*(bsize+2)+idx*size_batch:id*(bsize+2)+(idx+1)*size_batch+2]
            for i in range(len(x1[id])):
                output11_3[:, :, int(x1[id][i]), int(y1[id][i])] = 0
                output81[:, :, int(x1[id][i]), int(y1[id][i])] = 0
                output7[:, :, int(x1[id][i]), int(y1[id][i])] = 0

            ref_t_sol_1 = loss_func.get_local_Loss(output11_3)
            ref_t_sol_2 = loss_func.get_local_Loss(ref_local_sol[id*(bsize+2)+idx*size_batch:id*(bsize+2)+(idx+1)*size_batch+2, :, :, :])
            p_sim +=  mse_loss(output7[2:, :, :, :].to(accelerator.device), output11_3)
            p_local += (mse_loss(ref_t_sol_1, ref_t_sol_2[2:, :, :, :]))

            if epoch % 50 == 0 and id < coffe*ntb:
                if output_t == None:
                    output_t = output11_3[0:2].clone()
                elif idx == 0:
                    output_t = torch.concat((output_t, output11_3[0:2].clone()), dim=0).to(accelerator.device)

                output_t = torch.concat((output_t, output11_3[2:].clone()), dim=0).to(accelerator.device)

    loss += 1/(max(last_loss_weight,1)) * (p_sim + p_speed) + p_local

    return loss, p_sim, p_speed, p_local, output_t

def train(model, model1, input, n_iters, n_iters1, n_iters2, time_batch_size,
          dt, dx, num_time_batch, num, fre):
    state_detached1 = []
    prev_output1 = []

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x1 = np.load('./case/SeaFloor1/x1.npy')
    y1 = np.load('./case/SeaFloor1/y1.npy')
    # load previous9 model
    optimizer = optim.AdamW(model.parameters(), lr=1e-03)
    tt_flag = False
    scheduler = StepLR(optimizer, step_size=50000, gamma=0.975)

    optimizer_p = optim.AdamW(model1.parameters(), lr=1e-03)
    scheduler_p = StepLR(optimizer_p, step_size=50000, gamma=0.975)

    loss_func = loss_generator(num, dt, dx, fre)
    loss_func2 = loss_generator(num, dt, dx, fre)

    ref_speed = 1500*torch.ones((1,1,64*num,64*num)).to(accelerator.device)
    # ref_speed = torch.load('./case/SeaFloor1/ref_speed.pt').unsqueeze(dim=0).unsqueeze(dim=0)
    last_ref_speed = 1500 * torch.ones((1, 1, 64 * num, 64 * num)).to(accelerator.device)
    last_loss_weight = 1e8

    train_dataloader = DataLoader(input, time_batch_size+2, shuffle=False)

    train_dataloader, model, model1, loss_func, loss_func2, optimizer, optimizer_p, scheduler, scheduler_p = \
        accelerator.prepare(
            train_dataloader,
            model,
            model1,
            loss_func,
            loss_func2,
            optimizer,
            optimizer_p,
            scheduler,
            scheduler_p
        )

    for epoch in range(n_iters):
        batch_loss = 0.0
        batch_loss2 = 0.0
        batch_loss3 = 0.0
        batch_loss4 = 0.0
        flag_num = []
        history_loss = []
        temp_num = 1
        t_epoch = 0
        size_batch = 16
        for step, batch in enumerate(train_dataloader):
            flag_num.append(1)
        for i in range(batch.shape[0]):
            history_loss.append(1e25)
        epoch1 = 0
        for epoch1 in range(n_iters1):
            # input: [t,c,p,h,w]
            # update the first input for each time batch
            tc = 0
            alpha = 1
            output_t = None
            for step, batch in enumerate(train_dataloader):
                # update the first input for each time batch
                loc_x = x1[step]
                loc_y = y1[step]
                # if time_batch_id == 0:
                num_time_batch2 = int((batch.shape[0] - 2) / size_batch)

                for time_batch_id in range(flag_num[step] - 1, flag_num[step], 1):
                    ntb = flag_num[step] - 1
                    x_tt = input[ntb * size_batch + temp_num - 1: ntb * size_batch + temp_num].detach()
                    x_t = input[ntb * size_batch + temp_num: ntb * size_batch + temp_num + 1].detach()
                    # output is a list
                    ref_speed = ref_speed.detach()
                    output4, output1, output2, output3, second_last_state_forward, test_speed \
                        = model(ref_speed, size_batch, time_batch_id, loc_x, loc_y,
                            input, flag_num[step], temp_num, x_tt, x_t)

                    # get loss
                    # with torch.autograd.set_detect_anomaly(True):
                    loss, loss_local, loss_res, loss_local2, temp_num, input, t_flag, t_epoch, x_tt, x_t = compute_loss(
                    output1, output2, output3, loss_func,
                    step, time_batch_id, time_batch_size,
                    size_batch, tc, temp_num, input, history_loss, t_epoch, num_time_batch2,last_loss_weight)
                    if time_batch_id == flag_num[step] - 1:
                        t_loss = loss.item()
                        t_loss2 = loss_local.item()
                        t_loss3 = loss_res.item()
                        t_loss4 = loss_local2.item()
                    if t_flag:
                        batch_loss += loss.item()
                        batch_loss2 += loss_local.item()
                        batch_loss3 += loss_res.item()
                        batch_loss4 += loss_local2.item()

                    if time_batch_id == flag_num[step] - 1:
                        optimizer.zero_grad()
                        loss.backward()  # loss.backward()
                        optimizer.step()
                        scheduler.step()

                    if time_batch_id == flag_num[step] - 1 and flag_num[
                        step] < num_time_batch2 and time_batch_id != num_time_batch2 - 1:
                        if temp_num == size_batch + 1:
                            flag_num[step] += 1
                            temp_num = 1

                    elif time_batch_id == flag_num[step] - 1 and time_batch_id == num_time_batch2 - 1:
                        if temp_num == size_batch + 1:
                            print("Stop!")
                            tt_flag = True
                            break

                    train_dataloader = DataLoader(input, time_batch_size + 2, shuffle=False)
            t_epoch += 1
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("--------------------------------------------------------------------")
            print(f"epoch【{epoch + 1}】@{nowtime} sub_epoch【{epoch1 + 1}】flag_num {flag_num[0]} temp_num {temp_num} t_epoch {t_epoch}")
            print(f"loss= {t_loss:.2f}, loss_local= {t_loss2:.2f},loss_res={t_loss3:.2f}, loss_true={t_loss4:.2f}")
            print(f"bloss= {batch_loss:.2f}, bloss_local= {batch_loss2:.2f},bloss_res={batch_loss3:.2f}, bloss_true={batch_loss4:.2f}")
            if tt_flag:
                torch.save(input, './res/SeaFloor1/tensor_'+ str(epoch)+"_" + str(epoch1) + '.pt')
                tt_flag = False
                break

        output_t1 = input.clone().detach()
        if epoch!=0:
            last_loss_weight1 = batch_loss2
        else:
            last_loss_weight1 = 1e8

        epoch2 = 0
        if epoch!=0:
            n_iters2 = 100
        for epoch2 in range(n_iters2):
            ref_speed1 = model1(output_t1, time_batch_size, num_time_batch)
            loss_p, loss_1, loss_2, loss_3, output_t = \
                compute_loss_p(model, output_t1.clone().detach(), loss_func2, num_time_batch, ref_speed1, time_batch_size, last_loss_weight1, size_batch, last_ref_speed, tc, epoch)
            optimizer_p.zero_grad()
            loss_p.backward()
            optimizer_p.step()
            scheduler_p.step()
            ref_speed = ref_speed1.clone()
            last_loss_weight = loss_3.item()
            if (epoch2+1) % 50 == 0:
                cmap = cm.get_cmap('jet')
                plt.imshow(torch.squeeze(ref_speed, dim=1).detach().cpu().numpy().squeeze(), cmap=cmap)
                plt.colorbar()
                #plt.show()
                plt.savefig('./res/SeaFloor1/fig/speed_' + str(epoch) + "_" + str(epoch2) + '.png')
                plt.close()
                torch.save(ref_speed, './res/SeaFloor1/ref_speed/speed_' + str(epoch)+"_" + str(epoch2) +  '.pt')

            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("--------------------------------------------------------------------")
            print(f"epoch【{epoch + 1}】@{nowtime} sub_epoch【{epoch2 + 1}】 evaluate speed")
            print(f"loss_p = {loss_p.item():.2f}, loss_p_field = {loss_1:.2f}, loss_p_speed = {loss_2.item():.2f}, loss_p_ref = {loss_3.item():.2f}")
        last_ref_speed = ref_speed.clone().detach()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, scheduler, save_dir):
    '''save model and optimizer'''

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, save_dir)


def load_checkpoint(model, optimizer, scheduler, save_dir):
    '''load model and optimizer'''

    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    if (not optimizer is None):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print('Pretrained model loaded!')

    return model, optimizer, scheduler


def summary_parameters(model):
    for i in model.parameters():
        print(i.shape)


def frobenius_norm(tensor):
    return np.sqrt(np.sum(tensor ** 2))

if __name__ == '__main__':
    input_tensor = torch.load("./case/SeaFloor1/o_temp.pt")
    res1 = input_tensor.clone()

    time_steps = input_tensor.shape[0]

    sigmoid_n = -0.5
    dt = float(1 /4096.0)
    dx = 1.0
    fre = 25
    time_batch_size = 256
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    num_time_batch = int(time_steps / (time_batch_size + 2))
    n_iters_adam = 50
    n_iters_adam1 = 9999999
    n_iters_adam2 = 200
    pre_model_save_path = './checkpoint' \
                          '500.pt'
    model_save_path = './checkpoint1000.pt'
    fig_save_path = './figures/'
    n = 2

    model1 = PreNet(n, sigmoid_n).to(accelerator.device)

    model = PGONet(
        dt=dt,
        dx=dx,
        fre=fre).to(accelerator.device)

    start = time.time()
    train_loss = train(model, model1, input_tensor, n_iters_adam, n_iters_adam1, n_iters_adam2, time_batch_size,
                       dt, dx, num_time_batch, n, fre)
    end = time.time()

    np.save('./res/SeaFloor1/train_loss', train_loss)
    print('The training time is: ', (end - start))