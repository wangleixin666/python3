#!/usr/bin/env python    # -*- coding: utf-8 -*

from sklearn.datasets import load_iris
# 导入数据包
from sklearn.model_selection import train_test_split
# 用于数据分割
from sklearn.preprocessing import StandardScaler
# 用于标准化数据
from sklearn.neighbors import KNeighborsClassifier
# 导入KNN分类器模型
from sklearn.metrics import classification_report
# 导入进行详细的分析模型平均准确率，召回率，F1指标

iris = load_iris()

# print iris.data.shape
# print iris.DESCR
# 查看数据说明

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
# 标准化数据，注意X_train和X_test的转化函数不同

knc = KNeighborsClassifier()
knc.fit(X_train, Y_train)
"""训练KNC模型，进行参数估计????"""
# K近邻没有参数估计的过程，只是根据测试数据的分布直接做出分类决策
# 因此时间复杂度较高，内存消耗比较大

Y_predict = knc.predict(X_test)
# 用估计好的对测试样本进行预测

print 'the accuracy of KNClassifier is', knc.score(X_test, Y_test)
# 用自带的进行准确率的评估分析
print classification_report(Y_test, Y_predict, target_names=iris.target_names)