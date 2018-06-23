#!/usr/bin/env python    # -*- coding: utf-8 -*

from sklearn.datasets import load_boston
# 加载波士顿房价数据集
from sklearn.model_selection import train_test_split
# 引入数据分割的工具
from sklearn.preprocessing import StandardScaler
# 引入标准化数据的工具
from sklearn.neighbors import KNeighborsRegressor
# 引入K近邻回归模型
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train, Y_train.ravel())
uni_knr_Y_predict = uni_knr.predict(X_test)
# 使用预测方式为平均回归weights='uniform'

print 'R-squared of uniform-weighted KNeighborRegression:', uni_knr.score(X_test, Y_test)
print 'the mean squared of uniform-weighted KNeighborRegression:', mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(uni_knr_Y_predict))
print 'the mean absolute squared of uniform-weighted KNeighborRegression:', mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(uni_knr_Y_predict))

dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train, Y_train.ravel())
dis_knr_Y_predict = dis_knr.predict(X_test)
# 预测方式为根据距离加权回归weights='distance'

print 'R-squared of distance-weighted KNeighborRegression:', dis_knr.score(X_test, Y_test)
print 'the mean squared of distance-weighted KNeighborRegression:', mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(dis_knr_Y_predict))
print 'the mean absolute squared of distance-weighted KNeighborRegression:', mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(dis_knr_Y_predict))