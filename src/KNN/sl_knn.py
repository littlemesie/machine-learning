# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/3/19 23:25
@summary:
"""
from sklearn.neighbors import KNeighborsClassifier

# 加载数据
def load_data():
    train_data = []
    train_label = []
    with open('../../data/KNN/datingTestSet2.txt','r') as f:
        for line in f.readlines():
            line = line.strip()
            arr = line.split('\t')
            train_data.append(list(map(float, arr[0:3])))
            train_label.append(int(arr[-1]))
    return train_data, train_label

# 模型训练
def knn(train_data, train_label, test_data):
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(train_data, train_label)
    test_label = knn_clf.predict(test_data)
    print(test_label)

if __name__ == '__main__':
    train_data, train_label = load_data()
    test_data = [[38483, 10.273169, 1.808053]]
    print(test_data)
    # knn(train_data, train_label, test_data)