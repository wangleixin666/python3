#!/usr/bin/env python    # -*- coding: utf-8 -*

import pandas as pd
# 为了读取信息
from sklearn.model_selection import train_test_split
# 为了分割数据的测试集和训练集
from sklearn.feature_extraction import DictVectorizer
# 为了转化为特征向量
from sklearn.tree import DecisionTreeClassifier
# 导入决策树模型
from sklearn.metrics import classification_report
# 导入详细评估分析模型
from sklearn.ensemble import RandomForestClassifier
# 导入随机森林模型
from sklearn.ensemble import GradientBoostingClassifier
# 导入梯度上升模型

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# 从互联网直接读取titanic信息

# print titanic.info()
# 根据信息选取X.Y需要的特征，Y肯定是survived，X根据常识，选取pclass,age,sex三项

X = titanic[['pclass', 'age', 'sex']]
Y = titanic[['survived']]

# print X.info()
# print Y.info()
# 发现除了X中的age特征外，别的都是1313列，所以需要对age特征补全

X['age'].fillna(X['age'].mean(), inplace=True)
# print X.info()
# X.fillna('UNKNOWN', inplace=True)
# X._update_inplace(X['age'])
# fillna(): 返回一个数据拷贝，其中的缺失值都已被填充或者估算
# 利用633个age特征的平均值进行补全
# print X.info()
# 还是会提示警告A value is trying to be set on a copy of a slice from a DataFrame
"""
错误原因未知
test1可行，可能是因为数据格式问题？？？
"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=33)

"""需要把sex那种数据转化为数值特征向量"""
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

"""使用单一决策树进行训练，评估结果"""
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train.values.ravel())
dtc_Y_predict = dtc.predict(X_test)
print 'Accuracy of DecisionTreeClassifier is：', dtc.score(X_test, Y_test)
print classification_report(Y_test, dtc_Y_predict, target_names=['died', 'survived'])

"""使用随机森林分类器进行模型的集成，训练，预测，分析"""
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train.values.ravel())
rfc_Y_predict = rfc.predict(X_test)
print 'Accuracy of RandomForestClassifier is：', rfc.score(X_test, Y_test)
print classification_report(Y_test, rfc_Y_predict, target_names=['died', 'survived'])

"""使用梯度提升进行模型的集成，训练，预测，分析"""
gbc = GradientBoostingClassifier()
gbc.fit(X_train, Y_train.values.ravel())
gbc_Y_predict = gbc.predict(X_test)
print 'Accuracy of GradientBoostingClassifier is：', gbc.score(X_test, Y_test)
print classification_report(Y_test, gbc_Y_predict, target_names=['died', 'survived'])

"""
将fit中(X_train, Y_train)转化为(X_train, Y_train.values.ravel())
不然Y_train作为一个二维数组时会报错的
DataConversionWarning: 
A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
"""


"""
.values.ravel()和.ravel()的区别：

"""