# -*- coding:utf-8 -*-
from numpy import *
import Bayes
from Utils import Config

class Test(object):

    bayes = Bayes.Bayes()

    def testingNB(self):
        listOPosts, listClasses = self.bayes.loadDataSet()
        myVocabList = self.bayes.createVocabList(listOPosts)
        trainMat = []
        for postinDoc in listOPosts:
            trainMat.append(self.bayes.setOfWords2Vec(myVocabList, postinDoc))
        p0V, p1V, pAb = self.bayes.trainNB0(array(trainMat), array(listClasses))
        testEntry = ['love', 'my', 'dalmation']
        thisDoc = array(self.bayes.setOfWords2Vec(myVocabList, testEntry))
        testEntry = ['stupid', 'garbage']
        thisDoc = array(self.bayes.setOfWords2Vec(myVocabList, testEntry))
        print testEntry, 'classified as: ', self.bayes.classifyNB(thisDoc, p0V, p1V, pAb)

    # 测试函数
    def spamTest(self):
        docList = [];
        classList = [];
        fullText = []
        for i in range(1, 26):
            # 导入文件 并解析成词列表
            wordList = self.bayes.textParse(open(Config.DATAS + 'NaiveBayes/email/spam/%d.txt' % i).read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)
            wordList = self.bayes.textParse(open(Config.DATAS + 'NaiveBayes/email/ham/%d.txt' % i).read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
        vocabList = self.bayes.createVocabList(docList)  # create vocabulary
        trainingSet = range(50);
        testSet = []  # 随机构建测试函数
        for i in range(10):
            randIndex = int(random.uniform(0, len(trainingSet)))
            testSet.append(trainingSet[randIndex])
            del (trainingSet[randIndex])
        trainMat = [];
        trainClasses = []
        for docIndex in trainingSet:  # 测试分类器
            trainMat.append(self.bayes.bagOfWords2VecMN(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])
        p0V, p1V, pSpam = self.bayes.trainNB0(array(trainMat), array(trainClasses))
        errorCount = 0
        for docIndex in testSet:  # classify the remaining items
            wordVector = self.bayes.bagOfWords2VecMN(vocabList, docList[docIndex])
            if self.bayes.classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
                errorCount += 1
                print "classification error", docList[docIndex]
        e = float(errorCount) / len(testSet)
        return e
        print 'the error rate is: ', e

if __name__ == '__main__':
    # Test().testingNB()
    Test().spamTest()
