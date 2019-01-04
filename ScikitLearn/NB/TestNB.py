# -*- coding:utf-8 -*
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import numpy as np

def revs_data(arr):
    arr = np.array(arr)
    for x in range(0, arr.shape[0]):
        for y in range(0, arr.shape[1]):
            if arr[x, y] == "S":
                arr[x, y] = 0.1
            if arr[x, y] == "M":
                arr[x, y] = 0.2
            if arr[x, y] == "L":
                arr[x, y] = 0.3
    arr = arr.astype(float)
    return arr

def NB():
    gnb = GaussianNB()
    # iris = datasets.load_iris()
    # print iris.data
    # print type(iris.data)
    # y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
    # print("Number of mislabeled points out of a total %d points : %d"
    #        % (iris.data.shape[0],(iris.target != y_pred).sum()))
    # print iris

    list_X = [
                [1, 'S'],
                [1, 'M'],
                [1, 'M'],
                [1, 'S'],
                [1, 'S'],
                [2, 'S'],
                [2, 'M'],
                [2, 'M'],
                [2, 'L'],
                [2, 'L'],
                [3, 'L'],
                [3, 'M'],
                [3, 'M'],
                [3, 'L'],
                [3, 'L'],
            ]
    list_X = revs_data(list_X)
    print list_X
    list_Y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
    predict_X = revs_data([[1,'M']])

    y_pred = gnb.fit(list_X, list_Y).predict(predict_X)
    print y_pred

NB()
