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


        output1[n + 1, 0, :] = 0
        output1[n + 1, :, 0:1] = output1[n, :, 0:1] - dt * c[:, 0:1] * (output1[n, :, 0:1] - output1[n, :, 1:2]) / dx
        # for i in range(1, output1.shape[1]-1, 1):
        #     output1[n+1, i, output1.shape[1]- i] = output1[n, i - 1, output1.shape[1]-1- i]
        output1[n + 1, :, -1:] = output1[n, :, -1:] - dt * c[:, -1:] * (output1[n, :, -1:] - output1[n, :, -2:-1]) / dx
        output1[n + 1, -1, :] = output1[n, -1, ] - dt * c[-1, :] * (output1[n, -1:, :] - output1[n, -2:-1, :]) / dx
        # output1[:, -1, :] = 0
    print(sum)

    return torch.unsqueeze(output1, dim=1)

size= 256 + 2
dt = float(1/4096.0)
dx = 1
fre = 25
n = 2
Lx = Ly = 64*n # Length of the 2D domain

loc_x = [40*2,40*2,40*2,40*2,40*2,40*2,40*2,40*2,40*2,
         35*2,35*2,35*2,35*2,35*2,35*2,35*2,35*2,35*2,
         45*2,45*2,45*2,45*2,45*2,45*2,45*2,45*2,45*2]
loc_y = [4,10,40,48,64,80,110,116,122,
        4,16,32,48,64,80,96,112,124
        ,2,6,44,50,58,64,70,116,124]
np.save('./case/SeaFloor1/loc_x.npy', loc_x)
np.save('./case/SeaFloor1/loc_y.npy', loc_y)

tdx = 1  # 空间步长为0.01米
tdy = 1

c = 1500 * torch.ones((64*n, 64*n)).cuda()  # 生成一个波速为45的张量（可根据测试需要调整）
# for i in range(1, c.shape[1] - 1, 1):
#     c[i, c.shape[1] - i:] = 0
for i in range(64*n):
    if i<16*n:
        c[38*n+int(((16*n-i)**2)/((16*n) **2)*(15*n)):,i] = 0
    elif i<32*n:
        c[54*n-int(((32*n-i) ** 2) / ((16*n) ** 2) * (16*n)):, i] = 0
    elif i<48*n:
        c[40*n + int(((48*n - i) ** 2) / ((16*n) ** 2) * (14*n)):, i] = 0
    elif i<64*n:
        c[40*n + int(((i - 48*n) ** 2) / ((16*n) ** 2) * (20*n)):, i] = 0

cmap = cm.get_cmap('jet')
plt.imshow(c.detach().cpu().numpy().squeeze(), cmap=cmap)
plt.colorbar()
plt.savefig('./case/SeaFloor1/speed.png')
plt.show()

#设置时间步


#设置单点（单点声速预测）或多点声源（探测多障碍物）
x1=[[int(37.5*n),int(37.5*n),int(37.5*n),int(37.5*n),int(37.5*n)]]
y1=[[5*n,20*n,35*n,50*n,60*n]]
np.save('./case/SeaFloor1/x1.npy', x1)
np.save('./case/SeaFloor1/y1.npy', y1)
location = torch.ones((len(x1),2)).cuda()

output_s = torch.zeros((size*len(x1), 1, 64*n, 64*n)).cuda()
for i in range(len(x1)):
    # location[i,0]=x1[i]
    # location[i,1]=y1[i]
    u = np.zeros((2, 64*n, 64*n))
    output = torch.zeros((size, 64*n, 64*n)).cuda()
    x = np.arange(0, Lx, tdx)
    y = np.arange(0, Ly, tdy)
    X, Y = np.meshgrid(x, y)

    for idx in range(len(x1[i])):
        u[0, x1[i][idx], y1[i][idx]] = 1500 * np.sin(2 * 3.1415926 * fre * 0 * dt)
        u[1, x1[i][idx], y1[i][idx]] = 1500 * np.sin(2 * 3.1415926 * fre * 1 * dt)
    # u[0,:,:] += 10*np.exp(-((X - x1[i]) ** 2 + (Y - y1[i]) ** 2)/100)
    # u[1,:,:] += 10*np.exp(-((X - x1[i]) ** 2 + (Y - y1[i]) ** 2)/100)
    u = torch.from_numpy(u)
    output[0:2,:,:] = u[0:2,:,:]
    output=get_phy_Loss(output,c,size,x1[i],y1[i],dt,dx,fre)
    output_s[size*i:size*(i+1),:,:,:]=output[:,:,:,:].clone()

cmap = cm.get_cmap('jet')
plt.imshow(output_s[-1].detach().cpu().numpy().squeeze(), cmap=cmap)
plt.colorbar()
plt.show()

# ref_speed = torch.load('C:/Users/Xiarui/Downloads/OCA-NET-main/OCA-NET-main/res/speed_3050.pt')
# cmap = cm.get_cmap('jet')
# plt.imshow(torch.squeeze(ref_speed,dim=1).detach().cpu().numpy().squeeze(), cmap=cmap)
# plt.colorbar()
# plt.show()

# bsize = 16+2
# output_s1 = torch.zeros((output_s.shape[0]-2+2*int(output_s[2:].shape[0]/(bsize-2)),output_s.shape[1],output_s.shape[2],output_s.shape[3])).cuda()
# for i in range(int(output_s[2:].shape[0]/(bsize-2))):
#     output_s1[bsize*i:bsize*(i+1)] = output_s[(bsize-2)*i:2+(bsize-2)*(i+1)]
# dataloader = DataLoader(dataset=output_s1, batch_size=bsize)
# for data in dataloader:
# 	print(data.shape)
# 确保目录存在

# directory = './FFT/'
# if not os.path.exists(directory):
#     os.makedirs(directory)

# # 设定保存路径
# depth = 100
# num_files = int(output.shape[2] - depth/4)
# for i in range(1, num_files):
#     # 创建文件名，补全为四位数
#     filename = f"{i*4:d}.txt"
#     # 完整文件路径
#     filepath = os.path.join(directory, filename)
#
#     # 打开文件以写入数据（如果文件已存在，则覆盖其内容）
#     with open(filepath, 'w') as file:
#         # 写入数据到文件
#         for j in range(1024):
#             # 生成两个随机数并用空格分隔
#             data1 = 1 + j*float(1/1024)
#             data2 = output[1024+j,0,int(depth/4),i].data
#             # 写入一行数据
#             file.write(f"{data1} {data2}\n")

#保存文件
torch.save(c,'./case/SeaFloor1/ref_speed.pt')
print(output_s.shape)
torch.save(output_s[:],'./case/SeaFloor1/o_temp.pt')
# torch.save(location,'./case/location.pt')