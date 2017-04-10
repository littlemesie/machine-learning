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
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

# sigmoid 函数 inX 是一个向量
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 梯度上升法 dataMatIn是一个二维数组 每列代表不同的特征 每行代表不同的样本
# classLabels 是类别标签
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001  # 步伐
    maxCycles = 500 # 迭代次数
    weights = ones((n,1))
    for k in range(maxCycles):
        # 矩阵相乘
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)              # 错误率
        weights = weights + alpha * dataMatrix.transpose()* error # 梯度上升法计算
    return weights

# 最佳拟合直线图
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

#  随机梯度上升法
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights



dataMat,labelMat = loadDataSet()
# print data

w = gradAscent(dataMat,labelMat)
w1 = stocGradAscent0(array(dataMat),labelMat)
# plotBestFit(w1)
print w1
print w1