# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/3/12 22:22
@summary:
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def load_data_set(filename):
    dataset = []
    lableset = []
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            if not line:
                continue
            lineArr = []
            arr = line.split(',')
            for i, featrue in enumerate(arr):
                str_f = featrue.strip()
                if i == len(arr)-1:
                    if str_f == 'R':
                        lableset.append(1)
                    elif str_f == 'M':
                        lableset.append(2)
                else:
                    lineArr.append(float(str_f))
            dataset.append(lineArr)
    return dataset, lableset

def random_forest(dataset,lableset):
    rf = RandomForestClassifier(n_estimators=10)
    rf = rf.fit(dataset, lableset)
    score = np.mean(cross_val_score(rf, dataset, lableset, cv=5, scoring='roc_auc'))
    print(score)

if __name__ == '__main__':
    data_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"
    # 加载数据
    dataset, lableset = load_data_set(data_path + 'RandomForest/sonar-all-data.txt')
    print(dataset)
    random_forest(dataset, lableset)


