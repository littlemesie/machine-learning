# -*- coding:utf-8 -*-
'''
    PCA主成分分析方
'''

from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split('\t') for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat=2):
    """

    :param dataMat:
    :param topNfeat:
    :return: lowDDataMat 降维后的矩阵
    """
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # 去平均值
    # 计算协方差矩阵及其特征值
    covMat = cov(meanRemoved, rowvar=0)
    # eigVals特征值 eigVects特征向量
    eigVals,eigVects = linalg.eig(mat(covMat ))
    # 从小到大N个值排序（索引值）
    eigValInd = argsort(eigVals)

    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]

    # 将数据转换到新空间
    lowDDataMat = meanRemoved * redEigVects

    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def replaceNanWithMean():
    datMat = loadDataSet('testSet.txt', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat


datMat = replaceNanWithMean()
lowDDataMat, reconMat = pca(datMat)
print lowDDataMat

