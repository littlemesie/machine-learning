# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np


# 打开数据
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('horseColicTraining.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # dataMat.append([lineArr[0:-1]])
        # labelMat.append(int(lineArr[-1]))
    return dataMat,labelMat

dataMat,labelMat = loadDataSet()
print labelMat