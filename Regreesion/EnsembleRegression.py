#!/usr/bin/env python    # -*- coding: utf-8 -*

from sklearn.datasets import load_boston
# 加载波士顿房价数据集
from sklearn.model_selection import train_test_split
# 引入数据分割的工具
from sklearn.preprocessing import StandardScaler
# 引入标准化数据的工具
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
# 导入随机森林回归模型，极端随机森林，梯度提升回归模型
from sklearn.metrics import mean_absolute_error, mean_squared_error
# 引入数据评估模型
import numpy as np

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

rfr = RandomForestRegressor()
rfr.fit(X_train, Y_train.ravel())
rfr_Y_predict = rfr.predict(X_test)
# 使用随机森林回归模型

print 'R-squared of RandomForestRegressor:', rfr.score(X_test, Y_test)
print 'the mean squared of RandomForestRegressor:', mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(rfr_Y_predict))
print 'the mean absolute squared of RandomForestRegressor:', mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(rfr_Y_predict))

etr = ExtraTreesRegressor()
etr.fit(X_train, Y_train.ravel())
etr_Y_predict = etr.predict(X_test)
# 使用极端森林回归模型

print 'R-squared of ExtraTreesRegressor:', etr.score(X_test, Y_test)
print 'the mean squared of ExtraTreesRegressor:', mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(etr_Y_predict))
print 'the mean absolute squared of ExtraTreesRegressor:', mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(etr_Y_predict))

print np.sort(zip(etr.feature_importances_, boston.feature_names), axis=0)
# 利用训练好的极端回归森林模型，输出各种特征对预测目标的贡献度

gbr = GradientBoostingRegressor()
gbr.fit(X_train, Y_train.ravel())
gbr_Y_predict = gbr.predict(X_test)

print 'R-squared of GradientBoostingRegressor:', gbr.score(X_test, Y_test)
print 'the mean squared of GradientBoostingRegressor:', mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(gbr_Y_predict))
print 'the mean absolute squared of GradientBoostingRegressor:', mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(gbr_Y_predict))
