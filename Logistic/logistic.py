# -*- coding:utf-8 -*-

'''
    logistic算法
'''

from numpy import *

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
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

data,lables = loadDataSet()
print data