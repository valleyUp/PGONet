import torch
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.utils.data import Dataset, DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_phy_Loss(output1, c, size, loc_x, loc_y, dt, dx, fre):
    print(c)
    t_max = dt*(size-1)
    dy = dx
    r1, r2 = c ** 2 * dt ** 2 / dx ** 2, c ** 2 * dt ** 2 / dy ** 2
    sum = 0
    for n in range(1, int(t_max / dt)):
        # 在边界处设置固定边界条件
        #output1[:, 0, :] = output1[:, -1, :] = output1[:, :, 0] = output1[:, :, -1] = 0
        # 在内部节点上使用五点差分法计算新的波场
        output1[(n + 1), 1:-1, 1:-1] = 2 * output1[n, 1:-1, 1:-1] - output1[n - 1, 1:-1, 1:-1] + \
                                           r1[1:-1, 1:-1] * (output1[n, 2:, 1:-1] - 2 * output1[n, 1:-1, 1:-1] + output1[n,:-2, 1:-1]) + \
                                           r2[1:-1, 1:-1] * (output1[n, 1:-1, 2:] - 2 * output1[n, 1:-1, 1:-1] + output1[n,1:-1, :-2])
        for idx in range(len(loc_x)):
            output1[n + 1, loc_x[idx], loc_y[idx]] = 1500 * np.sin(2 * 3.1415926 * fre * (n + 1) * dt)


        output1[n + 1, 0, :] = output1[n, 0, ] - dt * c[0, :] * (output1[n, 0:1, :] - output1[n, 1:2, :]) / dx
        output1[n + 1, :, 0:1] = output1[n, :, 0:1] - dt * c[:, 0:1] * (output1[n, :, 0:1] - output1[n, :, 1:2]) / dx
        # for i in range(1, output1.shape[1]-1, 1):
        #     output1[n+1, i, output1.shape[1]- i] = output1[n, i - 1, output1.shape[1]-1- i]
        output1[n + 1, :, -1:] = output1[n, :, -1:] - dt * c[:, -1:] * (output1[n, :, -1:] - output1[n, :, -2:-1]) / dx
        output1[n + 1, -1, :] = output1[n, -1, ] - dt * c[-1, :] * (output1[n, -1:, :] - output1[n, -2:-1, :]) / dx
        #output1[n+1, -1, :] = 0
    print(sum)

    return torch.unsqueeze(output1, dim=1)



size=256+2
dt = float(1/4096)
dx = 1
fre = 25
n = 2
Lx = Ly = 64*n # Length of the 2D domain

tdx = 1  # 空间步长为0.01米
tdy = 1

c = 1500 * torch.ones((64*n, 64*n)).cuda()  # 生成一个波速为45的张量（可根据测试需要调整）
c[42:47,58:61]=340


cmap = cm.get_cmap('jet')
plt.imshow(c.detach().cpu().numpy().squeeze(), cmap=cmap)
plt.colorbar()
plt.savefig('./case/MultiObs3/speed.png')
plt.show()

#设置时间步


#设置单点（单点声速预测）或多点声源（探测多障碍物）
x1=[[10*n,10*n,50*n,50*n,32*n]]
y1=[[10*n,50*n,10*n,50*n,32*n]]
np.save('./case/MultiObs3/x1.npy', x1)
np.save('./case/MultiObs3/y1.npy', y1)
location = torch.ones((len(x1),2)).cuda()

output_s = torch.zeros((size*len(x1), 1, 64*n, 64*n)).cuda()
for i in range(len(x1)):
    u = np.zeros((2, 64*n, 64*n))
    output = torch.zeros((size, 64*n, 64*n)).cuda()
    x = np.arange(0, Lx, tdx)
    y = np.arange(0, Ly, tdy)
    X, Y = np.meshgrid(x, y)

    for idx in range(len(x1[i])):
        u[0, x1[i][idx], y1[i][idx]] = 1500 * np.sin(2 * 3.1415926 * fre * 0 * dt)
        u[1, x1[i][idx], y1[i][idx]] = 1500 * np.sin(2 * 3.1415926 * fre * 1 * dt)
    u = torch.from_numpy(u)
    output[0:2,:,:] = u[0:2,:,:]
    output=get_phy_Loss(output,c,size,x1[i],y1[i],dt,dx,fre)
    output_s[size*i:size*(i+1),:,:,:]=output[:,:,:,:].clone()

cmap = cm.get_cmap('jet')
plt.imshow(output_s[-1].detach().cpu().numpy().squeeze(), cmap=cmap)
plt.colorbar()
plt.show()


#保存文件
torch.save(c,'./case/MultiObs3/ref_speed.pt')
print(output_s.shape)
torch.save(output_s[:],'./case/MultiObs3/o_temp.pt')
