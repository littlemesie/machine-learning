# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

class Logistic(object):

    def iris(self):
        """使用iris数据测试"""
        X, y = load_iris(return_X_y=True)
        lr = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
        lr.fit(X, y)
        score = lr.score(X, y)
        print(score)

    def horse_colic(self):
        """"""
        train = pd.read_table('data/horseColicTraining.txt', header=None, sep='\t')
        train_x = train.drop(columns=21)
        train_y = train[21]
        lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        lr.fit(train_x, train_y)

        score = lr.score(train_x, train_y)
        print('train acc : {}'.format(score))

        test = pd.read_table('data/horseColicTest.txt', header=None, sep='\t')
        test_x = test.drop(columns=21)
        test_y = test[21]

        score = lr.score(test_x, test_y)
        print('test acc : {}'.format(score))

if __name__ == '__main__':
    logistic = Logistic()
    # logistic.iris()
    logistic.horse_colic()