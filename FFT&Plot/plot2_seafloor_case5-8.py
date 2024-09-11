import torch
import matplotlib.pyplot as plt
from matplotlib import cm

import matplotlib
import numpy as np

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 18}

matplotlib.rc('font', **font)

# 假设 file1.pt, file2.pt, file3.pt 已经被正确加载
# file1 = torch.load('D:/res/Boundary1/o_temp.pt')  # 假设这是一个张量或张量列表
# file2 = torch.load('D:/plotWaveEquation/tensor_10000_0.pt')
file1 = ['./Sea/1/ref_speed.pt','./Sea/2/ref_speed.pt','./Sea/3/ref_speed.pt','./Sea/4/ref_speed.pt']
file2 = ['./Sea/1/speed_49_99.pt','./Sea/2/speed_49_99.pt','./Sea/3/speed_49_99.pt','./Sea/4/speed_49_99.pt']
file3 = ['./Sea/1/x1.npy','./Sea/2/x1.npy','./Sea/3/x1.npy','./Sea/4/x1.npy']
file4 = ['./Sea/1/y1.npy','./Sea/2/y1.npy','./Sea/3/y1.npy','./Sea/4/y1.npy']
file5 = ['./Sea/1/loc_x.npy','./Sea/2/loc_x.npy','./Sea/3/loc_x.npy','./Sea/4/loc_x.npy']
file6 = ['./Sea/1/loc_y.npy','./Sea/2/loc_y.npy','./Sea/3/loc_y.npy','./Sea/4/loc_y.npy']
str=['a','b','c','d','e','f','g','h']
# 索引列表，跳过 64
indices = [72, 80, 88, 96, 108, 116, 126]
cmap = cm.get_cmap('jet')

# 创建一个图形和子图
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.3)  # 调整子图间距

norm = matplotlib.colors.Normalize(vmin=340,vmax=1500)

# 绘制每个子图并添加标题
for idx in range(8):
    ax = axs.flat[idx]  # 扁平化访问
    if idx < 4:  # 第一行
        img = torch.load(file1[idx]).cpu().numpy()  # 假设 file1 是张量列表或可以直接索引
        title = f'({str[idx]})Case {5+idx:.0f}'
    elif idx < 8:  # 第二行
        img = torch.load(file2[idx-4]).detach().cpu().numpy()
        title = f'({str[idx]})Predict Result of Case {1+idx:.0f}'

    # 假设 img 是二维的（例如灰度图像），如果是三维（例如 RGB），请相应调整
    im = ax.imshow(img.squeeze(), cmap=cmap, norm=norm)  # 使用 viridis 颜色映射
    if idx < 4:
        x1 = np.load(file3[idx])
        y1 = np.load(file4[idx])
        ax.scatter(y1, x1, label='Source', marker='o', c='r', s=25)
        x = np.load(file5[idx])
        y = np.load(file6[idx])
        ax.scatter(y, x, label='Sensor', marker='x', c='y', s=12)
        ax.legend()
    ax.set_title(title)
    ax.axis('on')  # 关闭坐标轴

# 创建颜色条
ax = axs.flatten()
cax = fig.add_axes([0.9, 0.1, 0.02, 0.8])  # 宽度, 高度, 左下角 x, 左下角 y
cbar = fig.colorbar(im, ax=[ax[i] for i in range(8)], cax = cax)
cbar.ax.set_ylabel('The Speed of Sound')  # 颜色条标签

plt.show()
