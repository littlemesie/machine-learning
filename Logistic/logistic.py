# -*- coding:utf-8 -*-

'''
    logistic算法
'''

from numpy import *
import matplotlib.pyplot as plt

class Logistic(object):

    def sigmoid(self,X):
        """
        sigmoid 函数，X 是一个向量
        :return:
        """
        return 1.0 / (1 + exp(-X))

    def grad_ascent(self, X, y):
        """
        梯度上升法
        :param X: 输入数据
        :param y: 输入分类（类别标签）
        :return:
        """
        data = mat(X)
        # transposeh函数numpy矩阵转置操作
        labels = mat(y).transpose()
        m, n = shape(data)
        alpha = 0.001  # 步伐
        max_cycles = 1000  # 迭代次数
        weights = ones((n, 1))
        for k in range(max_cycles):
            # 矩阵相乘
            h = self.sigmoid(data * weights)
            error = (labels - h)  # 错误率
            weights = weights + alpha * data.transpose() * error  # 梯度上升法计算
        return weights

    def random_grad_ascent(self, X, y, num=150):
        """
        随机梯度上升
        :param X: 输入数据
        :param y: 输入分类（类别标签）
        :return:
        """
        m, n = shape(X)
        weights = ones(n)
        for j in range(num):
            data_index = range(m)
            for i in range(m):
                alpha = 4 / (1.0 + j + i) + 0.0001
                rand_index = int(random.uniform(0, len(data_index)))
                h = sigmoid(sum(X[rand_index] * weights))
                error = y[rand_index] - h
                weights = weights + alpha * error * X[rand_index]
                del (X[rand_index])
        return weights

    def fit_predict(self, X, y=None):
        prob = sigmoid(sum(X * weights))
        if prob > 0.5:
            return 1.0
        else:
            return 0.0

# 打开数据
def loadDataSet():
    dataMat = []
    labelMat = []
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
    # transposeh函数numpy矩阵转置操作
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001  # 步伐
    maxCycles = 1000 # 迭代次数
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
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
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

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

# logistic分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt');
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))


dataMat,labelMat = loadDataSet()
dataArr = array(dataMat)


w = gradAscent(dataMat,labelMat)
w1 = stocGradAscent0(array(dataMat),labelMat)
# plotBestFit(w)
# print  'y = ' + '(-' + str(w[0]) + '-' + str(w[1]) + '*x)/' + str(w[2])
# print w1

# multiTest()