# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/3/23 14:33
@summary:
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def load_data(file_name):
    trian = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            cur_line = line.strip().split('\t')
            # 将数据转换为浮点类型,便于后面的计算
            flt_line = list(map(float, cur_line))  # 映射所有的元素为 float（浮点数）类型
            trian.append(flt_line)
    return trian

def k_means(data):
    km = KMeans(n_clusters=4)
    km.fit(data)
    km_pred = km.predict(data)
    centers = km.cluster_centers_

    # 可视化结果
    plt.scatter(np.array(data)[:, 1], np.array(data)[:, 0], c=km_pred)
    plt.scatter(centers[:, 1], centers[:, 0], c="r")
    plt.show()

if __name__ == '__main__':
    data = load_data('../../data/Kmeans/testSet.txt')
    k_means(data)
