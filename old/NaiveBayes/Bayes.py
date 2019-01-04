# -*- coding:utf-8 -*-
'''
    朴素贝叶斯分类器
'''
from numpy import *
import re
class Bayes(object):

    def loadDataSet(self):
        """
        @summary:创建实验样本
        :return:
        """
        postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                     ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                     ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                     ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                     ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                     ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        # 1代表侮辱性言论 0代表正常言论
        classVec = [0,1,0,1,0,1]
        return postingList,classVec

    def createVocabList(self,dataSet):
        """
        @summary: 创建一个包含文档所有词 但不重复的词向量 set数据类型
        :param dataSet:
        :return:
        """
        vocabSet = set([])  # 创建一个空集合
        for document in dataSet:
            vocabSet = vocabSet | set(document) # 创建两个集合的并集
        return list(vocabSet)

    def setOfWords2Vec(self,vocabList, inputSet):
        """
        @summary:
        :param vocabList: 词向量
        :param inputSet: 某个文档 输出文档向量
        :return:
        """
        # 创建一个都是0的向量
        returnVec = [0]*len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] = 1
            else:
                print "the word: %s is not in my Vocabulary!" % word
        return returnVec

    def train_matrix(self,postingList,vocabSet):
        """
        @summary:文档训练矩阵
        :param postingList:
        :param vocabSet:
        :return:
        """
        trainMat = []
        for postDoc in postingList:
            trainMat.append(self.setOfWords2Vec(vocabSet,postDoc))
        return trainMat

    def trainNB0(self,trainMatrix,trainCategory):
        """
        @summary:朴素贝叶斯分类器的训练函数 输入的是文档矩阵和标签向量
        :param trainMatrix:
        :param trainCategory:
        :return:
        """
        numTrainDocs = len(trainMatrix)
        numWords = len(trainMatrix[0])
        pAbusive = sum(trainCategory)/float(numTrainDocs)

        p0Num = ones(numWords)
        p1Num = ones(numWords)
        p0Denom = 2.0
        p1Denom = 2.0
        for i in range(numTrainDocs):
            if trainCategory[i] == 1:
                p1Num += trainMatrix[i]
                p1Denom += sum(trainMatrix[i])
            else:
                p0Num += trainMatrix[i]
                p0Denom += sum(trainMatrix[i])
        p1Vect = log(p1Num/p1Denom)          # 计算概率
        p0Vect = log(p0Num/p0Denom)
        return p0Vect,p1Vect,pAbusive

    def classifyNB(self,vec2Classify, p0Vec, p1Vec, pClass1):
        """
        @summary:要分类的向量
        :param vec2Classify:
        :param p0Vec:
        :param p1Vec:
        :param pClass1:
        :return:
        """
        p1 = sum(vec2Classify * p1Vec) + log(pClass1)
        p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
        if p1 > p0:
            return 1
        else:
            return 0

    def bagOfWords2VecMN(self,vocabList, inputSet):
        """
        @summary:词袋模型
        :param vocabList:
        :param inputSet:
        :return:
        """
        returnVec = [0]*len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] += 1
        return returnVec

    def textParse(self,bigString):
        """
        @summary: 分词
        :param bigString:
        :return:
        """
        listOfTokens = re.split(r'\W*', bigString)
        return [tok.lower() for tok in listOfTokens if len(tok) > 2]
