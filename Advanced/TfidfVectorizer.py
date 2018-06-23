# -*- coding: utf-8 -*

from sklearn.datasets import fetch_20newsgroups
# 导入新闻特征
from sklearn.model_selection import train_test_split
# 导入分割数据的工具包
from sklearn.feature_extraction.text import TfidfVectorizer
# 导入TfidfVectorizer工具进行文本特征提取
from sklearn.naive_bayes import MultinomialNB
# 导入朴素贝叶斯分类器
from sklearn.metrics import classification_report
# 导入精确的分类结果分析工具包

news = fetch_20newsgroups(subset='all')
# 即时下载新闻样本，subset='all'表示将尽2万条文本存储在news中

X_train, X_test, Y_train, Y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
# 对news中的数据data进行分割，25%用于测试

"""与之前的CountVectorizer的区别就是采用的预处理方法变成了TfidfVectorizer"""
tfidf_vec = TfidfVectorizer()
# 初始化，并且赋值给变量tfidf_vec

X_tfidf_train = tfidf_vec.fit_transform(X_train)
X_tfidf_test = tfidf_vec.transform(X_test)
# 使用频率统计的方法对原始数据转化为特征向量

"""接下来的贝叶斯分类过程用的是经过处理后的样本"""
mnb_tfidf = MultinomialNB()
# 初始化朴素贝叶斯分类器
mnb_tfidf.fit(X_tfidf_train, Y_train)
# 对进行完频率转化后的样本进行参数学习

print 'Accuracy of 20news using Native Bayes with TfidfVectorizer：', mnb_tfidf.score(X_tfidf_test, Y_test)
# 输出结果的准确度

Y_tfidf_predict = mnb_tfidf.predict(X_tfidf_test)
# 对结果进行预测

print classification_report(Y_test, Y_tfidf_predict, target_names=news.target_names)
# 输出其它的分类性能指标结果