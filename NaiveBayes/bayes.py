# -*- coding:utf-8 -*-
'''
    朴素贝叶斯分类器
'''
from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1代表侮辱性言论 0代表正常言论
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

# 创建一个包含文档所有词 但不重复的词向量 set数据类型
def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空集合
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 创建两个集合的并集
    return list(vocabSet)

# vocabList词向量 inputSet某个文档 输出文档向量
def setOfWords2Vec(vocabList, inputSet):
    # 创建一个都是0的向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

# 文档训练矩阵
def train_matrix(postingList,vocabSet):
    trainMat = []
    for postDoc in postingList:
        trainMat.append(setOfWords2Vec(vocabSet,postDoc))
    return trainMat

# 朴素贝叶斯分类器的训练函数 输入的是文档矩阵和标签向量
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0;
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

# vec2Classify要分类的向量 p0Vec, p1Vec, pClass1三个概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

testingNB()



postingList,classVec = loadDataSet()
vocabSet = createVocabList(postingList)
vec = setOfWords2Vec(vocabSet,postingList[0])
trainMat = train_matrix(postingList,vocabSet)
p0,p1,pA = trainNB0(trainMat,classVec)
# print vocabSet[26]
# print p0
# print p1

