import numpy as np
from scipy.special import jv
from scipy.special import hankel1
import math
import matplotlib.pyplot as plt
pi=3.1415926

k=2*pi*32/1500
p0=1*hankel1(0,k)/4
rs=math.sqrt(40*40+88*88)

theta0=math.atan(1)
thetas=math.atan(40/88)

fig,axs = plt.subplots(1,3)

x_asix=[]
res=[]

for num in range(0,128-50,1):
    x=num
    theta=math.atan(50/(128-x))
    r=math.sqrt(50*50+(128-x)*(128-x))
    p=1*pi/theta0
    a=0
    for n in range(0,50):
        a=a+math.sin((n-1/2)*pi/theta0*thetas)*math.sin((n-1/2)*pi/theta0*theta)*jv((n-1/2)*pi/theta0,k*min(rs,r))*hankel1((n-1/2)*pi/theta0,k*max(rs,r))
    p=a*p

    x_asix.append(x)
    p=20*math.log10(abs(p/p0))
    res.append(p)

f1 = open("./res/resT2.txt") # 返回一个文件对象
x11=[]
t1=[]
res1=[]
x3=[]
count10=0

line1 = f1.readline() # 调用文件的 readline()方法
while line1:
    t1.append(float(line1.split(" ")[0]))
    x_1 = float(line1.split(" ")[0])
    x11.append(20 * math.log10(float(line1.split(" ")[1]) /1500 / abs(p0)))

    #x3.append(abs(abs(x2)-abs(p))/abs(p0))
    line1 = f1.readline()

res_1 = np.array(res[1:])
x11_1 = np.array(x11)
print(np.mean(np.abs(res_1-x11_1)))
avg = np.mean(np.abs(res_1-x11_1))
print(len([num for num in np.abs(res_1-x11_1) if num < 5]))
#print(np.max(np.abs(res_1-x11_1)))
#print(np.min(np.abs(res_1-x11_1)))
#plt.scatter(t1,x1,label='FDM solution',marker='o',c='b',s=0.5)
axs[0].scatter(t1,x11,label='PGONet solution',marker='x',c='b',s=2.5)
axs[0].plot(x_asix, res, label='Analytical solution',c='r')
axs[0].set_xlim(0,78)
axs[0].set_xlabel('Distance(m)',fontsize=15)
axs[0].set_ylim(-80,80)
axs[0].set_ylabel('TL(db)',fontsize=15)

pi=3.1415926

k=2*pi*64/1500
p0=1*hankel1(0,k)/4
rs=math.sqrt(40*40+88*88)

theta0=math.atan(1)
thetas=math.atan(40/88)

x_asix=[]
res=[]

for num in range(0,128-45,1):
    x=num
    theta=math.atan(45/(128-x))
    r=math.sqrt(45*45+(128-x)*(128-x))
    p=1*pi/theta0
    a=0
    for n in range(0,60):
        a=a+math.sin((n-1/2)*pi/theta0*thetas)*math.sin((n-1/2)*pi/theta0*theta)*jv((n-1/2)*pi/theta0,k*min(rs,r))*hankel1((n-1/2)*pi/theta0,k*max(rs,r))
    p=a*p

    x_asix.append(x)
    p=20*math.log10(abs(p/p0))
    res.append(p)

f1 = open("./res/resT3.txt") # 返回一个文件对象
x11=[]
t1=[]
res1=[]
x3=[]
count10=0

line1 = f1.readline() # 调用文件的 readline()方法
while line1:
    t1.append(float(line1.split(" ")[0]))
    x_1 = float(line1.split(" ")[0])
    x11.append(20 * math.log10(float(line1.split(" ")[1]) / 1500 / abs(p0)))

    #x3.append(abs(abs(x2)-abs(p))/abs(p0))
    line1 = f1.readline()
res_1 = np.array(res[1:])
x11_1 = np.array(x11)
print(np.mean(np.abs(res_1-x11_1)))
avg = np.mean(np.abs(res_1-x11_1))
print(len([num for num in np.abs(res_1-x11_1) if num < 5]))
#print(np.max(np.abs(res_1-x11_1)))
#print(np.min(np.abs(res_1-x11_1)))
#plt.scatter(t1,x1,label='FDM solution',marker='o',c='b',s=0.5)
axs[1].scatter(t1,x11,label='PGONet solution',marker='x',c='b',s=0.5)
axs[1].plot(x_asix, res, label='Analytical solution',c='r')
axs[1].set_xlim(0,83)
axs[1].set_xlabel('Distance(m)',fontsize=15)
axs[1].set_ylim(-80,80)
axs[1].set_ylabel('TL(db)',fontsize=15)

pi=3.1415926

k=2*pi*64/1500
p0=1*hankel1(0,k)/4
rs=math.sqrt(30*30+98*98)

theta0=math.atan(1)
thetas=math.atan(30/98)

x_asix=[]
res=[]

for num in range(0,88,1):
    x=num
    theta=math.atan(40/(128-x))
    r=math.sqrt(40*40+(128-x)*(128-x))
    p=1*pi/theta0
    a=0
    for n in range(0,60):
        a=a+math.sin((n-1/2)*pi/theta0*thetas)*math.sin((n-1/2)*pi/theta0*theta)*jv((n-1/2)*pi/theta0,k*min(rs,r))*hankel1((n-1/2)*pi/theta0,k*max(rs,r))
    p=a*p

    x_asix.append(x)
    p=20*math.log10(abs(p/p0))
    res.append(p)

f1 = open("./res/resT4.txt") # 返回一个文件对象
x11=[]
t1=[]
res1=[]
x3=[]
count10=0

line1 = f1.readline() # 调用文件的 readline()方法
while line1:
    t1.append(float(line1.split(" ")[0]))
    x_1 = float(line1.split(" ")[0])
    x11.append(20 * math.log10(float(line1.split(" ")[1]) / 1500 / abs(p0)))

    #x3.append(abs(abs(x2)-abs(p))/abs(p0))
    line1 = f1.readline()
res_1 = np.array(res[1:])
x11_1 = np.array(x11)
print(np.mean(np.abs(res_1-x11_1)))
avg = np.mean(np.abs(res_1-x11_1))
print(len([num for num in np.abs(res_1-x11_1) if num < 5]))
#print(np.min(np.abs(res_1-x11_1)))
#plt.scatter(t1,x1,label='FDM solution',marker='o',c='b',s=0.5)
axs[2].scatter(t1,x11,label='PGONet solution',marker='x',c='b',s=0.5)
axs[2].plot(x_asix, res, label='Analytical solution',c='r')
axs[2].set_xlim(0,88)
axs[2].set_xlabel('Distance(m)',fontsize=15)
axs[2].set_ylim(-80,80)
axs[2].set_ylabel('TL(db)',fontsize=15)

axs[2].tick_params(labelsize=15)
axs[0].tick_params(labelsize=15)
axs[1].tick_params(labelsize=15)

axs[0].set_title('(a)Comparison of the frequency domain \n solution and the analytical solution of case2',fontproperties="Times New Roman",fontsize=20)
axs[1].set_title('(b)Comparison of the frequency domain \n solution and the analytical solution of case3',fontproperties="Times New Roman",fontsize=20)
axs[2].set_title('(c)Comparison of the frequency domain \n solution and the analytical solution of case4',fontproperties="Times New Roman",fontsize=20)

axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.show(block=True)