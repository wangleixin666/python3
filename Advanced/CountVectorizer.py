# -*- coding: utf-8 -*

from sklearn.datasets import fetch_20newsgroups
# 导入新闻特征
from sklearn.model_selection import train_test_split
# 导入分割数据的工具包
from sklearn.feature_extraction.text import CountVectorizer
# 导入CountVectorizer工具进行文本特征提取
from sklearn.naive_bayes import MultinomialNB
# 导入朴素贝叶斯分类器
from sklearn.metrics import classification_report
# 导入精确的分类结果分析工具包

news = fetch_20newsgroups(subset='all')
# 即时下载新闻样本，subser='all'表示将尽2万条文本存储在news中

X_train, X_test, Y_train, Y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
# 对news中的数据data进行分割，25%用于测试

"""其实就是在进行数据训练参数之前，对数据进行CountVectorizer的预处理"""
count_vec = CountVectorizer()
# 初始化，并且赋值给变量count_vec

X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)
# 使用频率统计的方法对原始数据转化为特征向量

"""接下来的贝叶斯分类过程用的是经过处理后的样本"""
mnb_count = MultinomialNB()
# 初始化朴素贝叶斯分类器
mnb_count.fit(X_count_train, Y_train)
# 对进行完频率转化后的样本进行参数学习

print 'Accuracy of 20news using Native Bayes with CountVectorizer：', mnb_count.score(X_count_test, Y_test)
# 输出结果的准确度

Y_count_predict = mnb_count.predict(X_count_test)
# 对结果进行预测

print classification_report(Y_test, Y_count_predict, target_names=news.target_names)
# 输出其它的分类性能指标结果