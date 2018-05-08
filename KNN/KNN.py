# -*- coding:utf-8 -*-
'''
    KNN算法
'''

from numpy import *
import operator
from Utils import Config

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels



# KNN算法
# inX 输入向量 dataSet 输入向量集，labels 标签向量；k：最近邻接点的个数
def classify(inX, dataSet, labels, k):
    # 获取行数
    dataSetSize = dataSet.shape[0]
    # 计算inX与dataSet的距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    # 矩阵的每一列相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # 返回从小到大的索引值
    sortedDistIndicies = distances.argsort()
    classCount={}
    # print  distances
    # 选择最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # 倒序排序 返回时频率发生最高的标签
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 处理文本数据 返回一个输入向量集和标签向量集
def file2matrix(filename):
    fr = open(filename)
    # 获取行数
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))

    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3] # 取每一行的前三个
        classLabelVector.append(int(listFromLine[-1]))
        # classLabelVector.append(listFromLine[-1]) # 取最后一列
        index += 1
    return returnMat,classLabelVector

# 做归一化处理
def autoNorm(dataSet):
    # 选取一列的最大值和最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

group,lables = createDataSet()
# print group
# print group[:2]

c = classify([1,1],group,lables,3)
# print c

returnData,classLabelVector = file2matrix(Config.DATAS + 'KNN/datingTestSet2.txt')
print autoNorm(returnData)
# print returnData
# print returnData[:,2]