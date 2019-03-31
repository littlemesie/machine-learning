# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/3/31 14:29
@summary:
"""
# GaussianNB_高斯朴素贝叶斯
import numpy as np
from sklearn.naive_bayes import GaussianNB

def test():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Y = np.array([1, 1, 1, 2, 2, 2])
    clf = GaussianNB()
    clf.fit(X, Y)
    print(clf.predict([[-0.8, -1]]))
    clf_pf = GaussianNB()
    clf_pf.partial_fit(X, Y, np.unique(Y))
    print(clf_pf.predict([[-0.8, -1]]))



if __name__ == '__main__':
    test()