# -*- coding:utf-8 -*-
"""
@summary:KNN算法
"""
from numpy import *
import operator
import matplotlib.pyplot as plt
from Utils import Config

class KNN(object):

    def createDataSet(self):
        """
        @summary:创建数据集和标签,用于测试
        :return:
        """
        group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
        labels = ['A','A','B','B']
        return group, labels

    def classify(self,inX, dataSet, labels, k):
        """
        @summary:KNN算法
        :param inX: 输入向量
        :param dataSet: 输入向量集
        :param labels: 标签向量
        :param k: 最近邻接点的个数
        :return:
        """
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
        # 选择最小的k个点
        for i in range(k):
            voteIlabel = labels[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        # 倒序排序 返回时频率发生最高的标签
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def file2matrix(self,filename):
        """
        @summary:处理文本数据 返回一个输入向量集和标签向量集
        :param filename:
        :return:
        """
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

    def autoNorm(self,dataSet):
        """
        @summary: 归一化处理
        :param dataSet:
        :return:
        """
        # 选取一列的最大值和最小值
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        ranges = maxVals - minVals
        normDataSet = zeros(shape(dataSet))
        m = dataSet.shape[0]
        normDataSet = dataSet - tile(minVals, (m,1))
        normDataSet = normDataSet/tile(ranges, (m,1))
        return normDataSet, ranges, minVals

    def createPlot(self):
        """
        @summary:分析test数据
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        datingDataMat, datingLabels = self.file2matrix(Config.DATAS + 'KNN/datingTestSet2.txt')
        # ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
        # 取第2列和第三列的数据
        ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
        ax.axis([-2, 25, -0.2, 2.0])
        plt.xlabel('Percentage of Time Spent Playing Video Games')
        plt.ylabel('Liters of Ice Cream Consumed Per Week')
        plt.show()

if __name__ == '__main__':
    knn = KNN()
    group,lables = knn.createDataSet()
    c = knn.classify([1,1],group,lables,3)
    print c
    knn.createPlot()
    returnData,classLabelVector = knn.file2matrix(Config.DATAS + 'KNN/datingTestSet2.txt')
    # print knn.autoNorm(returnData)