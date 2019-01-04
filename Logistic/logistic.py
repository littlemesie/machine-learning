# -*- coding:utf-8 -*-

'''
    logistic算法
'''

from numpy import *

class Logistic(object):

    def sigmoid(self,X):
        """
        sigmoid 函数，X 是一个向量
        :return:
        """
        return 1.0 / (1 + exp(-X))

    def grad_ascent(self, X, y, iter=1000):
        """
        梯度上升法
        :param X: 输入数据
        :param y: 输入分类（类别标签）
        :return:
        transposeh函数numpy矩阵转置操作
        """
        m, n = shape(X)
        alpha = 0.001  # 步伐
        weights = ones(n)
        for k in range(iter):
            for i in range(m):
                # 矩阵相乘
                h = self.sigmoid(sum(X[i] * weights))
                error = y[i] - h
                weights = weights + alpha * X[i] * error  # 梯度上升法计算
        return weights

    def random_grad_ascent(self, X, y, iter=150):
        """
        随机梯度上升
        :param X: 输入数据
        :param y: 输入分类（类别标签）
        :return:
        """
        m, n = shape(X)
        weights = ones(n)
        for j in range(iter):
            for i in range(m):
                alpha = 4 / (1.0 + j + i) + 0.0001
                h = self.sigmoid(sum(X[i] * weights))
                error = y[i] - h
                weights = weights + alpha * error * X[i]

        return weights

    def fit_predict(self, X, weights):
        prob = self.sigmoid(sum(X * weights))
        if prob > 0.5:
            return 1.0
        else:
            return 0.0


if __name__ == '__main__':
    lr = Logistic()
    train_data = []
    train_label = []
    with open('data/horseColicTraining.txt',encoding='utf-8') as f:
        for line in f.readlines():
            curr_line = line.strip().split('\t')
            line_arr = []
            for i in range(21):
                line_arr.append(float(curr_line[i]))
            train_data.append(line_arr)
            train_label.append(float(curr_line[21]))
    train_w = lr.random_grad_ascent(array(train_data), train_label, 1000)

    train_data = []
    train_label = []
    error_count = 0
    num_test = 0
    with open('data/horseColicTest.txt', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            num_test = i+1
            curr_line = line.strip().split('\t')
            line_arr = []
            for j in range(21):
                line_arr.append(float(curr_line[j]))
            if int(lr.fit_predict(array(line_arr), train_w)) != int(curr_line[21]):
                error_count += 1

    error_rate = (float(num_test-error_count) / num_test)
    print("the acc rate of this test is: %f" % error_rate)