# -*- coding:utf-8 -*-
'''
    预测泰坦尼克号生还
'''
import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt
from Utils import Config


train = pd.read_csv(Config.DATAS + 'Titanic/train.csv')
test = pd.read_csv(Config.DATAS + 'Titanic/test.csv')
print train.head(3)
all_data = [train,test]