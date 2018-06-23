#!/usr/bin/env python    # -*- coding: utf-8 -*

from sklearn.datasets import load_boston
# 加载波士顿房价数据集
from sklearn.model_selection import train_test_split
# 引入数据分割的工具
from sklearn.preprocessing import StandardScaler
# 引入标准化数据的工具
from sklearn.tree import DecisionTreeRegressor
# 导入回归树模型
from sklearn.metrics import mean_absolute_error, mean_squared_error
# 引入数据评估模型

boston = load_boston()

X = boston.data
Y = boston.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=33)

ss_X = StandardScaler()
ss_Y = StandardScaler()

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
Y_train = ss_Y.fit_transform(Y_train.reshape(-1, 1))
Y_test = ss_Y.transform(Y_test.reshape(-1, 1))
# 要把以为数据Y变成二维的

dtr = DecisionTreeRegressor()
dtr.fit(X_train, Y_train.ravel())
dtr_Y_predict = dtr.predict(X_test)
# 利用回归树模型

print 'R-squared value of DecisionTreeRegression:', dtr.score(X_test, Y_test)
print 'the mean squared error of DecisionTreeRegression:', mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(dtr_Y_predict))
print 'the mean absoluate error of DecisionTreeRegression:', mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(dtr_Y_predict))