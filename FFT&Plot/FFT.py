import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft


X=[]

filePath="./FFT4_1/" # 文件夹路径
fileList=os.listdir(filePath)

for file in fileList:
    f = open(os.path.join(filePath,file)) # 返回一个文件对象
    line = f.readline()  # 调用文件的 readline()方法
    s1 = float(file.split("_")[0][:-4])
    #s2 = float(file.split("_")[2][:-4])
    L=0
    while line:
        X.append(float(line.split(" ")[1]))
        line = f.readline()
        L=L+1

    Y = fft(X)
    p2 = np.abs(Y)  # 双侧频谱
    p1 = p2[:int(L / 2)]
    X = []
    # f = np.arange(int(L / 2)) * Fs / L
    print(str(s1)+" ", end="" )
    print(max(2*p1/L))
    # plt.plot(f, 2 * p1 / L)
    # plt.title('Single-Sided Amplitude Spectrum of X(t)')
    # plt.xlabel('f (Hz)')
    # plt.ylabel('|P1(f)|')
    # plt.show()
