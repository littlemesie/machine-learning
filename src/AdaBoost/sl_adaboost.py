# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/3/16 15:39
@summary:
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier


def load_data_set(filename):
    num_feat = len(open(filename).readline().split('\t'))
    data_arr = []
    label_arr = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat - 1):
            line_arr.append(float(cur_line[i]))
        data_arr.append(line_arr)
        label_arr.append(float(cur_line[-1]))
    return np.matrix(data_arr), label_arr

def adaboost(X, y):
    model = AdaBoostClassifier(n_estimators=10)
    model = model.fit(X, y)
    score = np.mean(cross_val_score(model, X, y, cv=5, scoring='roc_auc'))
    print(score)

if __name__ == '__main__':
    data_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"
    # 加载数据
    dataset, lableset = load_data_set(data_path + 'AdaBoost/horseColicTraining2.txt')
    adaboost(dataset, lableset)