import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn import metrics

def gbdt():
    train_data = pd.read_csv("../../data/GBDT/train.csv")

    X = train_data.drop(['Disbursed', 'ID'], axis=1)
    y = train_data['Disbursed']

    gbm = GradientBoostingClassifier(random_state=10)
    gbm.fit(X, y)
    y_pred = gbm.predict(X)
    y_predprob = gbm.predict_proba(X)[:, 1]
    print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

    # get_best1_param(X, y)
    # get_best2_param(X, y)
    # get_best3_param(X, y)

    gbm1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=7, min_samples_leaf=60,
                                      min_samples_split=1200, max_features='sqrt', subsample=0.8, random_state=10)
    gbm1.fit(X, y)
    y_pred = gbm1.predict(X)
    y_predprob = gbm1.predict_proba(X)[:, 1]
    print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

def get_best1_param(X, y):
    # 优化 从步长(learning rate)和迭代次数(n_estimators)入手
    param_test1 = {'n_estimators': range(20, 81, 10)}
    gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                                                 min_samples_leaf=20, max_depth=8, max_features='sqrt',
                                                                 subsample=0.8, random_state=10),
                            param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
    gsearch1.fit(X, y)
    print(gsearch1.best_params_, gsearch1.best_score_)

def get_best2_param(X, y):
    # 优化 从最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索
    param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 801, 200)}
    gsearch2 = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, min_samples_leaf=20,
                                             max_features='sqrt', subsample=0.8, random_state=10),
        param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
    gsearch2.fit(X, y)
    print(gsearch2.best_params_, gsearch2.best_score_)

def get_best3_param(X, y):
    # 优化 由于决策树深度7是一个比较合理的值，我们把它定下来，对于内部节点再划分所需最小样本数min_samples_split，我们暂时不能一起定下来，
    # 因为这个还和决策树其他的参数存在关联。下面我们再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参
    param_test3 = {'min_samples_split': range(800, 1900, 200), 'min_samples_leaf': range(60, 101, 10)}
    gsearch3 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=7,
                                                                 max_features='sqrt', subsample=0.8, random_state=10),
                            param_grid=param_test3, scoring='roc_auc', iid=False, cv=5)
    gsearch3.fit(X, y)
    print(gsearch3.best_params_, gsearch3.best_score_)


if __name__ == '__main__':
    gbdt()