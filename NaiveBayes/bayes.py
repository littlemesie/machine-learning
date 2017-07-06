# -*- coding:utf-8 -*-
'''
    朴素贝叶斯分类器
'''
from numpy import *
import re

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
        else:
            print "the word: %s is not in my Vocabulary!" % word
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

# 词袋模型 vocabList词向量 inputSet输入文档
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

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

# testingNB()


# 分词
def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# 测试函数
def spamTest():
    docList = [];
    classList = [];
    fullText = []
    for i in range(1, 26):
        # 导入文件 并解析成词列表
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    trainingSet = range(50);
    testSet = []  # 随机构建测试函数
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:  # 测试分类器
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error", docList[docIndex]
    e = float(errorCount) / len(testSet)
    return e
    print 'the error rate is: ', e



# error =  0
# for i in range(100):
#     error += spamTest()
# error = error / 100
# print error
postingList,classVec = loadDataSet()
vocabSet = createVocabList(postingList)
# vec = setOfWords2Vec(vocabSet,postingList[0])
vec = bagOfWords2VecMN(vocabSet,['love', 'my', 'dalmation'])
trainMat = train_matrix(postingList,vocabSet)
p0,p1,pA = trainNB0(trainMat,classVec)
# print vocabSet[26]
# print p0
# print vec

