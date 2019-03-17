# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
# Series 是创建一位数组
arr1 = np.arange(10)
print(arr1)

s1 = pd.Series(arr1)
print(s1)
print(s1[9])


arr2 = np.array(np.arange(12)).reshape(4,3)
print(arr2)
dic2 = {'a':[1,2,3,4],'b':[5,6,7,8],'c':[9,10,11,12],'d':[13,14,15,16]}
print(dic2)
print(type(dic2))
df2 = pd.DataFrame(dic2)
print(df2)
print(type(df2))
print(df2['a'])
dic3 = {'one':{'a':1,'b':2,'c':3,'d':4},'two':{'a':5,'b':6,'c':7,'d':8},'three':{'a':9,'b':10,'c':11,'d':12}}
df3 = pd.DataFrame(dic3)
print(df3)
print(type(df3))
print(df3['one']['b'])
# 添加索引
s4 = pd.Series(np.array([1,1,2,3,5,8]))
print(s4)
s4.index = ['a','b','c','d','e','f']
print(s4)