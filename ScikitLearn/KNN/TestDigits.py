# -*- coding:utf-8 -*-
"""
@summary:KNN算法实现手写字体识别
"""
from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
from Utils import Config
import KNN
from os import listdir

class TestDigits(object):

    def img2vector(self,filename):
        returnVect = np.zeros((1, 1024))
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
        trainingMat = np.zeros((m, 1024))
        for i in range(m):
            fileNameStr = trainingFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            hwLabels.append(classNumStr)
            trainingMat[i, :] = self.img2vector(Config.DATAS + 'KNN/digits/trainingDigits/%s' % fileNameStr)
        # 开始训练
        clf = NearestCentroid()
        clf.fit(trainingMat, hwLabels)

        testFileList = listdir(Config.DATAS + 'KNN/digits/testDigits')
        errorCount = 0.0
        mTest = len(testFileList)
        for i in range(mTest):
            fileNameStr = testFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            vectorUnderTest = self.img2vector(Config.DATAS + 'KNN/digits/testDigits/%s' % fileNameStr)
            classifierResult = clf.predict(vectorUnderTest)
            print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
            if (classifierResult != classNumStr): errorCount += 1.0
        print "\nthe total number of errors is: %d" % errorCount
        print "\nthe total error rate is: %f" % (errorCount / float(mTest))
        
if __name__ == '__main__':
    TestDigits().handwritingClassTest()