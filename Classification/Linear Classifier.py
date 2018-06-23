#!/usr/bin/env python    # -*- coding: utf-8 -*

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split 写法不对
from sklearn.preprocessing import StandardScaler
# 为了标准化数据
from sklearn.linear_model import LogisticRegression
# 逻辑斯蒂回归
from sklearn.linear_model import SGDClassifier
# 随机梯度参数估计
from sklearn.metrics import classification_report
# 该模块可以进行性能评测

column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
# 使用所有特征创建特征列表

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names=column_names)
# 从网站读取数据

data = data.replace(to_replace='?', value=np.nan)
# 替换?的缺失值为标准缺失值

data = data.dropna(how='any')
# 丢弃带有缺失值的数据

# print data.shape
# 输出data的数据量和维度

X_train, X_test, Y_train, Y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size=0.25, random_state=33)
# 随机选取25%用于测试，剩下的75%用于构造训练集合

# print Y_train.value_counts()
# print Y_test.value_counts()
# 查验样本的数量和类别分布

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
# 标准化数据，保证每个维度的特征数据方差为1，均值为0.
# 使得预测结果不会被某些维度过大的特征值而主导

lr = LogisticRegression()
sgdc = SGDClassifier(max_iter=5)
# 加上SGDC迭代次数，否则会警告
# 在0.19的版本中，SDGClassifier默认的迭代次数是5，0.21版本默认的迭代次数是1000

lr.fit(X_train, Y_train)
lr_y_predict = lr.predict(X_test)

sgdc.fit(X_train, Y_train)
sgdc_y_predict = sgdc.predict(X_test)
# 调用fit函数/模块来训练模型参数
# 使用训练好的模型lr对X_test进行预测，结果存储在lr_y_predict中

print 'Accuracy of LR Classifier:', lr.score(X_test, Y_test)
# 用模型自带的评分函数score获得模型在测试集上的准确性

print classification_report(Y_test, lr_y_predict, target_names=['Benign', 'Malignant'])
# 利用classification_report获得其他三个指标的结果，召回率、精确率、F1指标

print 'Accuracy of SGDC Classifier:', sgdc.score(X_test, Y_test)
print classification_report(Y_test, sgdc_y_predict, target_names=['Benign', 'Malignant'])
