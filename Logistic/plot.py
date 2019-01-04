# -*- coding:utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt
from Logistic import logistic
#
# 打开数据
def load_data_set():
    """打开数据"""
    data = []
    lables = []
    with open('data/testSet.txt',encoding='utf-8') as f:
        for line in f.readlines():
            line_arr = line.strip().split()
            data.append([1.0, float(line_arr[0]), float(line_arr[1])])
            lables.append(int(line_arr[2]))
    return data, lables

def plot_best_fit():
    """最佳拟合直线图"""
    lr = logistic.Logistic()
    data, lables = load_data_set()
    data = array(data)
    w = lr.grad_ascent(data, lables)
    weights = w.getA()
    n = shape(data)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(lables[i])== 1:
            xcord1.append(data[i,1])
            ycord1.append(data[i,2])
        else:
            xcord2.append(data[i,1])
            ycord2.append(data[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    # 拟合一条最佳直线
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

plot_best_fit()