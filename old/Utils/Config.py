# -*- coding:utf-8 -*-
"""
@summary:定义一些常量
"""

import platform

if 'Windows' in platform.system():

    root = 'E:/python/machine-learning/'
else:
    root = '/Users/mesie/Pycharm/machine-learning/'

# 数据集目录
DATAS = root + 'data/'