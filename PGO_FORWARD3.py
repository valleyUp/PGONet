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

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)

lapl_op2 = [[[[    0,   1, 0],
             [    1,   -4,   1],
             [0, 1,    0]]]]
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


class PGONet(nn.Module):
    ''' physics-informed convolutional-recurrent neural networks '''

    def __init__(self, dt, dx, fre):

        super(PGONet, self).__init__()
        #self.laplace_conv = LaplaceConv2d()
        self.fre = fre
        # input channels of layer includes input_channels and hidden_channels of cells
        self.dt = dt
        self.dx = dx

        # ConvLSTM(Forward)
        self.input_layer9 = weight_norm(nn.Conv2d(3, 1, kernel_size=(3,3), stride=1,
                                                padding=1))


        self.ref_sol = torch.load('./case/Forward2/o_temp.pt').cuda()
        self.test_ref_speed = torch.load('./case/Forward2/ref_speed.pt').unsqueeze(dim=0).unsqueeze(dim=0).cuda()
        self.apply(initialize_weights)

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
        x_tt = batch[ntb *bsize+step:ntb *bsize+step+1].detach()
        x_t = batch[ntb *bsize+step+1:ntb *bsize+step+2].detach()
        x_t4 = torch.zeros_like(x_t).cuda()
        x_t4[:, :, 1:-1, 1:-1] = ((2 * x_t[:, :, 1:-1, 1:-1].detach() - x_tt[:, :, 1:-1, 1:-1].detach() ) +
                                  (x_t[:,:, 2:, 1:-1].detach()  - 4 * x_t[:,:, 1:-1, 1:-1].detach()  + x_t[:,:,:-2, 1:-1].detach()
                                   + x_t[:,:, 1:-1, 2:].detach()  + x_t[:,:,1:-1, :-2].detach() ) * (
                    ref_speed[:, :, 1:-1, 1:-1].detach()  ** 2) * (self.dt ** 2) / (self.dx ** 2))
        outputs3.append(x_t4.clone())
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
                                             ref_speed[:, :, 1:-1, 1:-1].detach() ** 2) * (self.dt ** 2) / (
                                                 self.dx ** 2))
        outputs2.append(x_temp7.clone())
        x_temp7[:, :, 0, :] = 0
        x_temp7[:, :, :, 0:1] = x_t[:, :, :, 0:1] - self.dt * ref_speed[:, :, :, 0:1].detach() * (
                x_t[:, :, :, 0:1] - x_t[:, :, :, 1:2]) / self.dx
        for i in range(1, x_temp7.shape[2] - 1, 1):
            x_temp7[:, :, i, x_temp7.shape[2] - i] = x_t[:, :, i - 1, x_temp7.shape[2] - 1 - i]
        x_temp7[:, :, :, -1:] = x_t[:, :, :, -1:] - self.dt * ref_speed[:, :, :, -1:].detach() * \
                                (x_t[:, :, :, -1:] - x_t[:, :, :, -2:-1]) / self.dx
        x_temp7[:, :, -1, :] = 0
        for idx in range(len(loc_x)):
            if loc_x[idx] != -1:
                x_temp7[:, :, int(loc_x[idx]), int(loc_y[idx])] = self.ref_sol[id * bsize + step + 2:id * bsize + step + 3,
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

        self.ref_sol = torch.load('./case/Forward2/o_temp.pt').cuda()
        self.num = num
        self.dttt = dtt
        self.dxx = dxx
        self.fre = fre

def compute_loss(output71, output2, output3, loss_func, id, id2,
                 bsize, bsize1, coffe, flag_num,
                 batch,history_loss, t_epoch,num_batch_size2):
    ''' calculate the phycis loss '''

    mse_loss = nn.MSELoss(reduction='mean')
    x_tt = output71[-2:-1].clone().detach()
    x_t = output71[-1:].clone().detach()
    t_flag = False
    i = flag_num - 1
    p_local = mse_loss(
        loss_func.ref_sol[id * (bsize + 2) + id2 * bsize1 + 2+i:id * (bsize + 2) + id2 * bsize1 + 3+i, :, :, :],
        output71[2:3, :, :, :])
    t_loss = mse_loss(output2[2:3,:,1:-1,1:-1],output3[0:1,:,1:-1,1:-1].detach())

    if t_loss  < history_loss[id * (bsize + 2) + id2 * bsize1 + 2 + i]:
        history_loss[id * (bsize + 2) + id2 * bsize1 + 2 + i] = t_loss
        if id2 >= coffe * num_batch_size2:
            batch[id * (bsize + 2) + id2 * bsize1 + 2 + i:id * (bsize + 2) + id2 * bsize1 + 3 + i] =output71[2: 3]

    if i==flag_num-1 and (((t_epoch>=3000 or t_loss< 1e-4) and (id2!=0 or i!=0)) or t_epoch == 30000):
        flag_num += 1
        t_flag =True
        t_epoch = 0

    print(p_local.item()," ",t_loss.item())
    return t_loss, p_local, 0, p_local, flag_num, batch, t_flag, t_epoch, x_tt, x_t

def train(model, input, n_iters, time_batch_size,
          dt, dx, num, fre):
    state_detached1 = []
    prev_output1 = []
    tc = 0
    alpha = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x1 = np.load('./case/Forward2/x1.npy')
    y1 = np.load('./case/Forward2/y1.npy')
    # load previous9 model
    optimizer = optim.AdamW(model.parameters(), lr=1e-03)
    scheduler = StepLR(optimizer, step_size=50000, gamma=0.975)
    tt_flag = False
    loss_func = loss_generator(num, dt, dx, fre)

    ref_speed = torch.load('./case/Forward2/ref_speed.pt').cuda()
    ref_speed = ref_speed.unsqueeze(dim=0).unsqueeze(dim=0)
    train_dataloader = DataLoader(input, time_batch_size+2, shuffle=False)

    flag_num = []
    history_loss = []
    temp_num = 1
    t_epoch = 0
    size_batch = 64
    for step, batch in enumerate(train_dataloader):
        flag_num.append(1)
    for i in range(batch.shape[0]):
        history_loss.append(1e25)

    input1 = input.clone()
    for epoch in range(n_iters):
        # input: [t,c,p,h,w]
        if epoch % 50 == 0:
            output = None
        # update the first input for each time batch

        batch_loss = 0.0
        batch_loss2 = 0.0
        batch_loss3 = 0.0
        batch_loss4 = 0.0
        #size_batch = 16
        for step, batch in enumerate(train_dataloader):
            # update the first input for each time batch
            loc_x = x1[step]
            loc_y = y1[step]
            # if time_batch_id == 0:
            num_time_batch2 = int((batch.shape[0]-2)/size_batch)
            for time_batch_id in range(flag_num[step]-1,flag_num[step],1):
                if time_batch_id == 0 and temp_num-1 == 0:
                    x_tt = batch[0:1, :, :, :].detach()
                    x_t = batch[1:2, :, :, :].detach()
                else:
                    hidden_state1 = state_detached1

                # output is a list
                ref_speed = ref_speed.detach()
                output4, output1, output2, output3, second_last_state_forward, test_speed \
                    = model(ref_speed, size_batch, time_batch_id, loc_x, loc_y,
                            input, flag_num[step], temp_num, x_tt, x_t)

                # get loss
                # with torch.autograd.set_detect_anomaly(True):

                loss, loss_local2, loss_local3, loss_true, temp_num, input, t_flag, t_epoch, x_tt, x_t = compute_loss(
                    output1, output2, output3, loss_func,
                    step, time_batch_id, time_batch_size,
                    size_batch, tc, temp_num, input, history_loss, t_epoch, num_time_batch2)
                if time_batch_id == flag_num[step]-1:
                    batch_loss += loss.item()
                    batch_loss2 += loss_local2.item()
                    batch_loss3 += loss_local3
                    batch_loss4 += loss_true.item()
                else:
                    batch_loss += loss
                    batch_loss2 += loss_local2
                    batch_loss3 += loss_local3
                    batch_loss4 += loss_true

                if time_batch_id == flag_num[step]-1:
                    optimizer.zero_grad()
                    loss.backward()  # loss.backward()
                    optimizer.step()
                    scheduler.step()

                if time_batch_id == flag_num[step]-1 and flag_num[step] < num_time_batch2 and time_batch_id != num_time_batch2-1:
                    if temp_num == size_batch+1:
                        flag_num[step] += 1
                        temp_num = 1

                elif time_batch_id == flag_num[step] - 1 and time_batch_id == num_time_batch2 - 1:
                    if temp_num == size_batch + 1:
                        print("Stop!")
                        tt_flag = True
                        break

                if t_flag:
                    state_detached1 = []
                    for i in range(len(second_last_state_forward)):
                        (h, c) = second_last_state_forward[i]
                        state_detached1.append((h, c))
                train_dataloader = DataLoader(input, time_batch_size + 2, shuffle=False)
        if epoch % 50000 == 0 and epoch != 0:
            torch.save(input, './res/tensor_forward2/tensor_' + str(epoch) + '_' + str(step) + '.pt')
        t_epoch += 1
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("--------------------------------------------------------------------")
        print(f"epoch【{epoch + 1}】@{nowtime} flag_num {flag_num[0]} temp_num {temp_num} t_epoch {t_epoch}")
        print(f"loss= {batch_loss:.2f}, loss_res= {batch_loss2:.2f},loss_local={batch_loss3:.2f}, loss_true={batch_loss4:.2f}, alpha={alpha:.3f}")
        if tt_flag:
            torch.save(input, './res/tensor_forward2/tensor_' + str(epoch) + '_' + str(step) + '.pt')
            break

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


if __name__ == '__main__':
    input_tensor = torch.load("./case/Forward2/o_temp.pt")
    res1 = input_tensor.clone()
    time_steps = input_tensor.shape[0]

    sigmoid_n = -0.5
    dt = float(1/(8192*2))
    dx = 1
    fre = 64
    time_batch_size = 4096
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    num_time_batch = int(time_steps / (time_batch_size + 2))
    n_iters_adam = 10000000001
    pre_model_save_path = './checkpoint' \
                          '500.pt'
    model_save_path = './checkpoint1000.pt'
    fig_save_path = './figures/'
    n = 2
    pre_model_save_path = './checkpoint' \
                          '500.pt'
    model_save_path = './checkpoint1000.pt'
    fig_save_path = './figures/'
    n = 2
    model = PGONet(
        dt=dt,
        dx=dx,
        fre=fre).cuda()
    start = time.time()
    train_loss = train(model, input_tensor, n_iters_adam, time_batch_size, dt, dx, n, fre)
    end = time.time()

    np.save('./res/tensor_forward2/train_loss', train_loss)
    print('The training time is: ', (end - start))