# -*- coding:utf-8 -*-
'''
    预测泰坦尼克号生还
'''
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
all_data = [train_data,test_data]
# pclass = train_data[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
#
# sex = train_data[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)
#
# sibsp = train_data[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)
#
# parch = train_data[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print sibsp

for data in all_data:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
#
# sibsp = train_data[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print sibsp

for data in all_data:
    data['IsAlone'] = 0
    data.loc[data['FamilySize']  ==1,'IsAlone']= 1


# isalone = train_data[['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean()
# print isalone

g = sb.FacetGrid(train_data,col='Survived')
g.map(plt.hist,'Age',bins=20)
