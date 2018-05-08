# -*- coding:utf-8 -*-

from numpy import *
import KNN
import matplotlib.pyplot as plt
from Utils import Config
fig = plt.figure()
ax = fig.add_subplot(111)
datingDataMat,datingLabels = KNN.file2matrix(Config.DATAS + 'KNN/datingTestSet2.txt')
#ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
# 取第2列和第三列的数据
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
ax.axis([-2,25,-0.2,2.0])
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Liters of Ice Cream Consumed Per Week')
plt.show()
