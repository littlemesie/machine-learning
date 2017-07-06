# -*- coding:utf-8 -*-
'''
    企业信用等级分类
'''

import numpy as np
from sklearn.cluster import KMeans
import sys

# 加载数据
def loadDataSet(fileName):
    comName = []
    dataMat = []
    labelMat = []
    lineArr = open(fileName).readlines()

    for i in range(1, len(lineArr)):
    # for i in range(1,100):
        line = lineArr[i].strip().split(',')
        comName.append(line[0])

        dataMat.append([float(line[j] == line[j] and line[j] or 0) for j in range(1, len(line)-2)])
        labelMat.append(float(line[len(line)-1]))

    return comName, dataMat, labelMat

comName, dataMat, labelMat = loadDataSet('data.txt')

km = KMeans(n_clusters=5)
label = km.fit_predict(dataMat)

comCluster = [[],[],[],[],[],[]]
for i in range(len(label)):
    if label[i] == 4:
        print comName[i] + ':' +'A' + '得分' + str(labelMat[i])
    elif label[i] == 3:
        print comName[i] + ':' + 'B' + '得分' + str(labelMat[i])
    elif label[i] == 2:
        print comName[i] + ':' + 'C' + '得分' + str(labelMat[i])
    elif label[i] == 1:
        print comName[i] + ':' + 'D' + '得分' + str(labelMat[i])
    else:
        print comName[i] + ':' + 'E' + '得分' + str(labelMat[i])