import torch
import matplotlib.pyplot as plt
from matplotlib import cm

import matplotlib
import numpy as np

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 12}

matplotlib.rc('font', **font)

# 假设 file1.pt, file2.pt, file3.pt 已经被正确加载
file1 = torch.load('./Compare/o_temp.pt')[2:]  # 假设这是一个张量或张量列表
file2 = torch.load('./Compare/u_res2_20000_PhyCRNet.pt')[1:]
file3 = torch.load('./Compare/u_res2_50000_FNO.pt')[1:]
file4 = torch.load('./Compare/u_res2_50000_CNO.pt')[1:]
file5 = torch.load('./Compare/u_res2_50000_KNO.pt')[1:]
file6 = torch.load('./Compare/u_res2_50000_DeepONet.pt')[1:]
# file1 = torch.load('./forward2.pt')  # 假设这是一个张量或张量列表
# file2 = torch.load('./tensor_10000_0_T2.pt')
# 索引列表，跳过 64
indices = [15, 23, 31, 39, 47, 55, 63]
cmap = cm.get_cmap('jet')

# 创建一个图形和子图
fig, axs = plt.subplots(6, 7, figsize=(20, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.3)  # 调整子图间距

norm = matplotlib.colors.Normalize(vmin=-250,vmax=250)

# 绘制每个子图并添加标题
for idx in range(42):
    ax = axs.flat[idx]  # 扁平化访问
    if idx < 7:  # 第一行
        img = file1[indices[idx]].cpu().numpy()  # 假设 file1 是张量列表或可以直接索引
        title = f'FDM, {(indices[idx])*float(1/25):.3f}s'
    elif idx < 14:  # 第二行
        img = file2[indices[idx-7]].detach().cpu().numpy()
        title = f'PhyCRNet, {(indices[idx-7])*float(1/25):.3f}s'
    elif idx < 21:  # 第二行
        img = file3[indices[idx-14]].detach().cpu().numpy()
        title = f'FNO, {(indices[idx-14])*float(1/25):.3f}s'
    elif idx < 28:  # 第二行
        img = file4[indices[idx-21]].detach().cpu().numpy()
        title = f'CNO, {(indices[idx-21])*float(1/25):.3f}s'
    elif idx < 35:  # 第二行
        img = file3[indices[idx-28]].detach().cpu().numpy()
        title = f'KNO, {(indices[idx-28])*float(1/25):.3f}s'
    elif idx < 42:  # 第二行
        img = file4[indices[idx-35]].detach().cpu().numpy()
        title = f'DeepONet, {(indices[idx-35])*float(1/25):.3f}s'

    # 假设 img 是二维的（例如灰度图像），如果是三维（例如 RGB），请相应调整
    im = ax.imshow(img.squeeze(), cmap=cmap, norm=norm)  # 使用 viridis 颜色映射
    ax.set_title(title)
    ax.axis('off')  # 关闭坐标轴

# 创建颜色条
ax = axs.flatten()
cax = fig.add_axes([0.9, 0.1, 0.02, 0.8])  # 宽度, 高度, 左下角 x, 左下角 y
cbar = fig.colorbar(im, ax=[ax[i] for i in range(14)], cax = cax)
cbar.ax.set_ylabel('Pressure')  # 颜色条标签

plt.show()