# -*- coding:utf-8 -*-
'''
    房屋价格预测
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

train = pd.read_csv('/Users/mesie/Pycharm/data/house/train.csv')
test = pd.read_csv('/Users/mesie/Pycharm/data/house/test.csv')
# combine = [train, test]
# print combine

# 计算与SalePrice相关列的相关性
cor = train.corr()['SalePrice']
# c = corr[np.argsort(corr)[::-1]]

# 画出相关性
num_feat = train.columns[train.dtypes != object]
num_feat = num_feat[1:-1]
labels = []
values = []
for col in num_feat:
    labels.append(col)
    values.append(np.corrcoef(train[col].values, train.SalePrice.values)[0, 1])

ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(10, 10))
rects = ax.barh(ind, np.array(values), color='red')
ax.set_yticks(ind + ((width) / 2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation Coefficients w.r.t Sale Price")
# plt.show()

# 计算所有的相关性
correlations=train.corr()
attrs = correlations.iloc[:-1,:-1]

threshold = 0.5
important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]).unstack().dropna().to_dict()

unique_important_corrs = pd.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])),
        columns=['Attribute Pair', 'Correlation'])

    # sorted by absolute value
unique_important_corrs = unique_important_corrs.ix[
    abs(unique_important_corrs['Correlation']).argsort()[::-1]]


corrMatrix=train[["SalePrice","OverallQual","GrLivArea","GarageCars",
                  "GarageArea","GarageYrBlt","TotalBsmtSF","1stFlrSF","FullBath",
                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]].corr()

sns.set(font_scale=1.10)
plt.figure(figsize=(10, 10))

sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between features')

