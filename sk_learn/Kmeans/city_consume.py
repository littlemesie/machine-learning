# -*- coding:utf-8 -*-
'''
 聚类全国城市消费水平
'''

import numpy as np
from sklearn.cluster import KMeans

def loadData(filePath):
    fr = open(filePath,'r+')
    lines = fr.readlines()
    data = []
    cityName = []
    for line in lines:
        items = line.strip().split(",")
        cityName.append(items[0])
        data.append([float(items[i]) for i in range(1, len(items))])
        # for i in range(1,len(items)):
        #     data.append([float(items[i])])
    return data, cityName

data, cityName = loadData('city.txt')
km = KMeans(n_clusters=3)
label = km.fit_predict(data)
expenses = np.sum(km.cluster_centers_,axis=1)
CityCluster = [[],[],[],[]]
for i in range(len(cityName)):
    CityCluster[label[i]].append(cityName[i])
for i in range(len(CityCluster)):
    # print("Expenses:%.2f" % expenses[i])
    print(CityCluster[i])
