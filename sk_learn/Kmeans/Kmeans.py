# -*- coding:utf-8 -*-
'''
 图像分割
'''
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans

def loadData(filePath):
    f = open(filePath,'rb') # 以二进制的形式打开图片
    data = []
    img = image.open(f) # 以列表形式返回图片的像素值
    m,n = img.size
    # 将每一个像素点处理成0-1范围内 返回保存在data中
    for i in range(m):
        for j in range(n):
            x,y,z = img.getpixel((i,j))
            data.append([x/256.0,y/256.0,z/256.0])

    f.close()
    return np.mat(data),m,n


imgData,row,col = loadData('bull.jpg')
label = KMeans(n_clusters=4).fit_predict(imgData)

label = label.reshape([row, col])
pic_new = image.new("L", (row, col))
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))
pic_new.save("result-bull-4.jpg", "JPEG")

print imgData