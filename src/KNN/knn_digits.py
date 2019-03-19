# -*- coding:utf-8 -*-
"""
@summary:KNN算法实现手写字体识别
"""
from numpy import *
from Utils import Config
import KNN
from os import listdir

class TestDigits(object):

    def img2vector(self,filename):
        returnVect = zeros((1, 1024))
        fr = open(filename)
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32 * i + j] = int(lineStr[j])
        return returnVect

    def handwritingClassTest(self):
        hwLabels = []
        # 加载训练数据集
        trainingFileList = listdir(Config.DATAS + 'KNN/digits/trainingDigits')
        m = len(trainingFileList)
        trainingMat = zeros((m, 1024))
        for i in range(m):
            fileNameStr = trainingFileList[i]
            fileStr = fileNameStr.split('.')[0]  # take off .txt
            classNumStr = int(fileStr.split('_')[0])
            hwLabels.append(classNumStr)
            trainingMat[i, :] = self.img2vector(Config.DATAS + 'KNN/digits/trainingDigits/%s' % fileNameStr)
        testFileList = listdir(Config.DATAS + 'KNN/digits/testDigits')  # iterate through the test set
        errorCount = 0.0
        mTest = len(testFileList)
        for i in range(mTest):
            fileNameStr = testFileList[i]
            fileStr = fileNameStr.split('.')[0]  # take off .txt
            classNumStr = int(fileStr.split('_')[0])
            vectorUnderTest = self.img2vector(Config.DATAS + 'KNN/digits/testDigits/%s' % fileNameStr)
            classifierResult = KNN.KNN().classify(vectorUnderTest, trainingMat, hwLabels, 3)
            print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
            if (classifierResult != classNumStr): errorCount += 1.0
        print("\nthe total number of errors is: %d" % errorCount)
        print("\nthe total error rate is: %f" % (errorCount / float(mTest)))
        
if __name__ == '__main__':
    TestDigits().handwritingClassTest()