import torch
import numpy as np

temp = torch.load("./o_temp.pt").cuda()
print(temp.shape)

x_i = torch.ones((126 * 126 * 1, 3)).cuda()
u_i = torch.ones((126 * 126 * 1, 1)).cuda()
for i in range(1):
    for j in range(126):
        for k in range(126):
            x_i[i * 126 * 126 + j * 126 + k, 2] = i * float(1/25)
            x_i[i * 126 * 126 + j * 126 + k, 0] = (j+1)*100
            x_i[i * 126 * 126 + j * 126 + k, 1] = (k+1)*100
            u_i[i * 126 * 126 + j * 126 + k, 0] = temp[0:1, :, j+1, k+1 ].item()

torch.save(x_i,'./x_i.pt')
torch.save(u_i,'./u_i.pt')

# x_f = torch.ones((62 * 62 * 32, 3)).cuda()
# for i in range(32):
#     for j in range(62):
#         for k in range(62):
#             x_f[i * 62 * 62 + j * 62 + k, 2] = (i+1)*0.01
#             x_f[i * 62 * 62 + j * 62 + k, 0] = j+1
#             x_f[i * 62 * 62 + j * 62 + k, 1] = k+1
#
# x_f = torch.ones((20 * 32, 3)).cuda()
# true_f = torch.ones((20 * 32, 1)).cuda()
# for i in range(32):
#     for j in range(10):
#         x_f[i * 20 + j, 2] = (i + 1) * 0.01
#         x_f[i * 20 + j, 0] = (j + 1) * 6
#         x_f[i * 20 + j, 1] = 0
#         true_f[i * 20 + k, 0] = temp[i + 2:i + 3, :, (j + 1) * 6,0:1].item()
#
#     for k in range(10):
#         x_f[i * 20 + 10 + k, 2] = (i+1)*0.01
#         x_f[i * 20 + 10 + k, 0] = 0
#         x_f[i * 20 + 10 + k, 1] = (k+1)*6
#         true_f[i * 20 + 10 + k, 0] = temp[i+2:i+3,:,0:1,(k+1)*6].item()
#
# torch.save(x_f,'./x_f.pt')
# torch.save(true_f,'./t_f.pt')

# x_f = torch.ones((126 * 126 * 64, 3)).cuda()
# true_f = torch.ones((126 * 126 * 64, 1)).cuda()
# for i in range(64):
#     for j in range(126):
#         for k in range(126):
#             x_f[i * 126 * 126 + j * 126 + k, 2] = (i + 1) * float(1/100)
#             x_f[i * 126 * 126 + j * 126 + k, 0] = j + 1
#             x_f[i * 126 * 126 + j * 126 + k, 1] = k + 1
#             true_f[i * 126 * 126 + j * 126 + k, 0] = temp[i + 2:i + 3, :, j+1, k + 1].item()
#
# torch.save(x_f,'./x_f.pt')
# torch.save(true_f,'./t_f.pt')


x_b = torch.ones((4 * 128 * 64, 3)).cuda()
for i in range(64):
    for j in range(128):
        for k in range(4):
            x_b[i * 128 * 4 + j * 4 + k, 2] = (i+1)*float( 1/25)
            if k == 0:
                x_b[i * 128 * 4 + j * 4 + k, 0] = 0
                x_b[i * 128 * 4 + j * 4 + k, 1] = j*100
            if k == 1:
                x_b[i * 128 * 4 + j * 4 + k, 0] = 127*100
                x_b[i * 128 * 4 + j * 4 + k, 1] = j*100
            if k == 2:
                x_b[i * 128 * 4 + j * 4 + k, 0] = j*100
                x_b[i * 128 * 4 + j * 4 + k, 1] = 0
            if k == 3:
                x_b[i * 128 * 4 + j * 4 + k, 0] = j*100
                x_b[i * 128 * 4 + j * 4 + k, 1] = 127*100

#torch.save(x_f,'./x_f.pt')
torch.save(x_b,'./x_b.pt')

# x_i = torch.ones((64 * 64 * 1, 3)).cuda()
# for i in range(1):
#     for j in range(64):
#         for k in range(64):
#             x_i[i * 64 * 64 + j * 64 + k, 2] = (i+1)*0.01
#             x_i[i * 64 * 64 + j * 64 + k, 0] = j+1
#             x_i[i * 64 * 64 + j * 64 + k, 1] = k+1
# torch.save(x_i,'./x_i.pt')

# x_t = torch.ones((64 * 64 * 1, 1)).cuda()
# init = torch.load("E:/Image_Dataset/True/temp10.pt").cuda()
# for i in range(1):
#     for j in range(64):
#         for k in range(64):
#             x_t[i * 64 * 64 + j * 64 + k, 0] = init[i,:,j,k]
# torch.save(x_t,'./x_t.pt')

# torch.save(x_i,'./x_i.pt')