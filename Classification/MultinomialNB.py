#!/usr/bin/env python    # -*- coding: utf-8 -*

from sklearn.datasets import fetch_20newsgroups
# 从sklearn.datasets中导入新闻数据抓取器
from sklearn.model_selection import train_test_split
# 导入分割数据
from sklearn.feature_extraction.text import CountVectorizer
# 导入用于文本特征向量转化模块
from sklearn.naive_bayes import MultinomialNB
# 导入朴素贝叶斯模型
from sklearn.metrics import classification_report
# 用来精确分析模型的平均准确率，召回率，F1指标

news = fetch_20newsgroups(subset='all')
# 需要即时从互联网下载数据

# print len(news.data)
# print news.data[0]
# 查验数据规模和细节

X_train, X_test, Y_train, Y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
# 分割数据25%用于测试，75用于训练

vec = CountVectorizer()
# 引入特征向量转化
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)
"""没有标准化数据，与其他几种分类器不同？？？"""
# 因为是特征向量提取，而不是简单地数据标准化

mnb = MultinomialNB()
# 引入朴素贝叶斯模型进行预测，初始化
mnb.fit(X_train, Y_train)
# 利用训练数据进行参数估计
Y_predict = mnb.predict(X_test)
# 用训练好的模型对测试数据进行预测，结果存在Y_predict中

# 接下来对模型准确性进行评估
print 'the accuracy of Native Bayes Classifier is', mnb.score(X_test, Y_test)
print classification_report(Y_test, Y_predict, target_names=news.target_names)
