# -*- coding:utf-8 -*-

'''
    logistic算法
'''

import sys
from numpy import *
import matplotlib.pyplot as plt

# sys.setrecursionlimit(10000)

# 打开数据
def loadDataSet():
    data = []; lables= []
    file = open('testSet.txt')
    for line in file.xreadlines():
        lineArr = line.strip().split()
        data.append([1.0,float(lineArr[0]),float(lineArr[1])])
        lables.append(int(lineArr[2]))
    return data,lables

# sigmoid 函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 梯度上升法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500 # 迭代次数
    weights = ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        # 矩阵相乘
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error # 梯度上升法计算
    return weights

# 画图
def plotBestFit(w):
    weights = w.getA()
    # print w
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    # 拟合一条最佳直线
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()


data,lables = loadDataSet()
# print data

w = gradAscent(data,lables)

plotBestFit(w)
#print w