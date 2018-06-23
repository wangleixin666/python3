# -*- coding: utf-8 -*

from sklearn.datasets import fetch_20newsgroups
# 导入新闻特征
from sklearn.model_selection import train_test_split
# 导入分割数据的工具包
from sklearn.feature_extraction.text import CountVectorizer
# 导入CountVectorizer工具进行文本特征提取
from sklearn.feature_extraction.text import TfidfVectorizer
# 导入TfidfVectorizer工具进行文本特征提取
from sklearn.naive_bayes import MultinomialNB
# 导入朴素贝叶斯分类器
from sklearn.metrics import classification_report
# 导入精确的分类结果分析工具包

news = fetch_20newsgroups(subset='all')
# 即时下载新闻样本，subser='all'表示将尽2万条文本存储在news中

X_train, X_test, Y_train, Y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
# 对news中的数据data进行分割，25%用于测试

"""
使用停用词过滤配置初始化
也就是在每条文本都出现的常用词汇
比如英文中的the，a，以黑名单方式过滤掉，提高模型性能
"""
count_filter_vec, tfidf_fileter_vec = CountVectorizer(analyzer='word', stop_words='english'), TfidfVectorizer(analyzer='word', stop_words='english')

# 对样本进行量化处理
X_count_filter_train = count_filter_vec.fit_transform(X_train)
X_count_filter_test = count_filter_vec.transform(X_test)

# 配置贝叶斯分类器
mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(X_count_filter_train, Y_train)

print 'Accuracy of 20news using Native Bayes with CountVectorizer：', mnb_count_filter.score(X_count_filter_test, Y_test)
# 输出结果的准确度

Y_count_filter_predict = mnb_count_filter.predict(X_count_filter_test)
# 对结果进行预测

print classification_report(Y_test, Y_count_filter_predict, target_names=news.target_names)
# 输出其它的分类性能指标结果

X_tfidf_filter_train = tfidf_fileter_vec.fit_transform(X_train)
X_tfidf_filter_test = tfidf_fileter_vec.transform(X_test)

# 配置贝叶斯分类器
mnb_tfidf_filter = MultinomialNB()
mnb_tfidf_filter.fit(X_tfidf_filter_train, Y_train)

print 'Accuracy of 20news using Native Bayes with TfidfVectorizer：', mnb_tfidf_filter.score(X_tfidf_filter_test, Y_test)
# 输出结果的准确度

Y_tfidf_filter_predict = mnb_count_filter.predict(X_tfidf_filter_test)
# 对结果进行预测

print classification_report(Y_test, Y_tfidf_filter_predict, target_names=news.target_names)
# 输出其它的分类性能指标结果
