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
        for i in range(1, output1.shape[1]-1, 1):
            output1[n+1, i, output1.shape[1]- i] = output1[n, i - 1, output1.shape[1]-1- i]
        output1[n + 1, :, -1:] = output1[n, :, -1:] - dt * c[:, -1:] * (output1[n, :, -1:] - output1[n, :, -2:-1]) / dx
        #output1[n + 1, -1, :] = output1[n, -1, ] - dt * c[-1, :] * (output1[n, -1:, :] - output1[n, -2:-1, :]) / dx
        output1[:, -1, :] = 0
    print(sum)

    return torch.unsqueeze(output1, dim=1)

size= 4096+2
dt = float(1/(8192*2))
dx = 1
fre = 32
n = 2
Lx = Ly = 64*n # Length of the 2D domain

tdx = 1  # 空间步长为0.01米
tdy = 1

c = 1500 * torch.ones((64*n, 64*n)).cuda()  # 生成一个波速为45的张量（可根据测试需要调整）
for i in range(1, c.shape[1] - 1, 1):
    c[i, c.shape[1] - i:] = 0
# for i in range(64*n):
#     if i<16*n:
#         c[50*n-int(((16*n-i)**2)/((16*n) **2)*(10*n)):,i] = 0
#     elif i<32*n:
#         c[50*n+int(((i-16*n) ** 2) / ((16*n) ** 2) * (10*n)):, i] = 0
#     elif i<48*n:
#         c[40*n + int(((48*n - i) ** 2) / ((16*n) ** 2) * (20*n)):, i] = 0
#     elif i<64*n:
#         c[40*n + int(((i - 48*n) ** 2) / ((16*n) ** 2) * (20*n)):, i] = 0

# cmap = cm.get_cmap('jet')
# plt.imshow(c.detach().cpu().numpy().squeeze(), cmap=cmap)
# plt.colorbar()
# plt.show()
# 以下为多障碍物测试时生成代码，可根据需要调整
# x2 = torch.arange(0, Lx, dx).cuda()
# y2 = torch.arange(0, Ly, dy).cuda()
# X2, Y2 = torch.meshgrid(x2, y2, indexing='ij')
# l=5
# s_x=15
# s_y=15
# s_u = (2 * l) ** 2 / ((X2 - s_x) ** 2 + (Y2 - s_y) ** 2 + 1)
# c=c-s_u
# c[0:2,:]=340
# c[23:30,13:19]=340
# c[68:75,71:79]=340
# c[26:32,102:110]=340
# c[115:122,45:52]=340

cmap = cm.get_cmap('jet')
plt.imshow(c.detach().cpu().numpy().squeeze(), cmap=cmap)
plt.colorbar()

#设置时间步


#设置单点（单点声速预测）或多点声源（探测多障碍物）
x1=[[int(40/dx)]]
y1=[[int(40/dx)]]
# np.save('./case/Forward/x1.npy', x1)
# np.save('./case/Forward/y1.npy', y1)
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

output = torch.load("./tensor_34095_0_T2.pt")
# func = torch.nn.MSELoss()
# print(func(output,output_s))
directory = './FFT2/'
directory_t = './FFT2_1/'
if not os.path.exists(directory):
    os.makedirs(directory)

# 设定保存路径
depth = int(50/dx)
print(depth)
num_files = int((output.shape[2] - depth))
for i in range(1, num_files):
    # 创建文件名，补全为四位数
    filename = f"{i:d}.txt"
    # 完整文件路径
    filepath = os.path.join(directory, filename)
    # 打开文件以写入数据（如果文件已存在，则覆盖其内容）
    with open(filepath, 'w') as file:
        # 写入数据到文件
        for j in range(4096-2048):
            # 生成两个随机数并用空格分隔
            data1 = 2048/(8192*2) + j*float(1/(8192*2))
            data2 = output_s[2048+2+j,0,int(depth/dx),i].data
            # 写入一行数据
            file.write(f"{data1} {data2}\n")

    filepath_1 = os.path.join(directory_t, filename)
    with open(filepath_1, 'w') as file1:
        # 写入数据到文件
        for j in range(4096-2048):
            # 生成两个随机数并用空格分隔
            data1 = 2048/(8192*2) + j*float(1/(8192*2))
            data2 = output[2048+2+j,0,int(depth),i].data
            # 写入一行数据
            file1.write(f"{data1} {data2}\n")

#保存文件
# torch.save(c,'./case/Forward/ref_speed.pt')
# print(output_s.shape)
torch.save(output_s[:],'./Forward1.pt')
# torch.save(output_s[:],'./case/Forward/o_temp.pt')
# torch.save(location,'./case/location.pt')