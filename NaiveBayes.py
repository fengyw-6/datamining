# encoding:utf-8

import json

import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self):
        self.model = {}  # key 为类别名 val 为字典PClass表示该类的概率，PFeature:{}对应对于各个特征的概率

    def calEntropy(self, y):  # 计算熵
        valRate = y.value_counts().apply(lambda x: x / y.size)  # 频次汇总 得到各个特征对应的概率
        valEntropy = np.inner(valRate, np.log2(valRate)) * -1
        return valEntropy

    def fit(self, xTrain, yTrain=pd.Series()):  # pd.Series()创建一个空的一维数组
        '''
        输入测试数据和分类标签
        :param xTrain:测试数据
        :param yTrain:分类标签，默认为最后一列
        :return:无返回值
        '''
        if not yTrain.empty:  # 如果不为空，将标签连接到数据的最后一列
            xTrain = pd.concat([xTrain, yTrain], axis=1)  # xTrain, yTrain竖直方向上连接 https://www.jianshu.com/p/421f040dfe2f
        self.model = self.buildNaiveBayes(xTrain)
        return self.model

    def buildNaiveBayes(self, xTrain):
        '''
        创建贝叶斯分类器
        :param xTrain:训练数据
        :return:
        '''
        yTrain = xTrain.iloc[:, -1]  # 取xTrain的最后一列（分类标签）
        yTrainCounts = yTrain.value_counts()  # 频次汇总 得到各个特征对应的概率
        yTrainCounts = yTrainCounts.apply(lambda x: (x + 1) / (yTrain.size + yTrainCounts.size))  # 使用了拉普拉斯平滑
        retModel = {}
        for nameClass, val in yTrainCounts.items():
            retModel[nameClass] = {'PClass': val, 'PFeature': {}}  # PClass表示该类的概率，PFeature:{}对应对于各个特征的概率
        propNamesAll = xTrain.columns[:-1]  # 取除最后一列之外其他列的列标签
        allPropByFeature = {}
        for nameFeature in propNamesAll:
            allPropByFeature[nameFeature] = list(xTrain[nameFeature].value_counts().index) # 统计每一列中的所有可能取值
        #print(allPropByFeature)
        #print(pd.DataFrame(xTrain.groupby(xTrain.columns[-1])))
        for nameClass, group in xTrain.groupby(xTrain.columns[-1]):
            for nameFeature in propNamesAll:
                eachClassPFeature = {}
                propDatas = group[nameFeature]
                propClassSummary = propDatas.value_counts()  # 频次汇总 得到各个特征对应的概率
                for propName in allPropByFeature[nameFeature]:
                    if not propClassSummary.get(propName):
                        propClassSummary[propName] = 0  # 如果有属性灭有，那么自动补0
                Ni = len(allPropByFeature[nameFeature])
                propClassSummary = propClassSummary.apply(lambda x: (x + 1) / (propDatas.size + Ni))  # 使用了拉普拉斯平滑
                for nameFeatureProp, valP in propClassSummary.items():
                    eachClassPFeature[nameFeatureProp] = valP
                retModel[nameClass]['PFeature'][nameFeature] = eachClassPFeature
        return retModel

    def predictBySeries(self, data):
        curMaxRate = None
        curClassSelect = None
        for nameClass, infoModel in self.model.items():
            rate = 0
            rate += np.log(infoModel['PClass'])
            PFeature = infoModel['PFeature']

            for nameFeature, val in data.items():
                propsRate = PFeature.get(nameFeature)
                if not propsRate:
                    continue
                rate += np.log(propsRate.get(val, 0))  # 使用log加法避免很小的小数连续乘，接近零
                # print(nameFeature, val, propsRate.get(val, 0))
            # print(nameClass, rate)
            if curMaxRate == None or rate > curMaxRate:
                curMaxRate = rate
                curClassSelect = nameClass

        return curClassSelect

    def predict(self, data):
        if isinstance(data, pd.Series):
            return self.predictBySeries(data)
        return data.apply(lambda d: self.predictBySeries(d), axis=1)


def createDataset():
    dataSet = [
        ['green', 'suo', 'zhuo', 'qing', 'ao', 'ying', 'yes'],
        ['black', 'suo', 'chen', 'qing', 'ao', 'ying', 'yes'],
    ]
    return dataSet


if __name__ == '__main__':
    dataTrain = pd.read_csv("car - 副本.csv",encoding="gbk")  # encoding声明编码方式，此处可以删除
    # print(type(dataTrain))
    naiveBayes = NaiveBayes()  # 声明class实体，用于调用方法
    treeData = naiveBayes.fit(dataTrain)

    print(json.dumps(treeData, ensure_ascii=False))

    pd = pd.DataFrame({'预测值': naiveBayes.predict(dataTrain), '正取值': dataTrain.iloc[:, -1]})
    print(pd)
    print('正确率:%f%%' % (pd[pd['预测值'] == pd['正取值']].shape[0] * 100.0 / pd.shape[0]))
