#!/usr/bin/env python    # -*- coding: utf-8 -*

from sklearn.datasets import load_boston
# 加载波士顿房价数据集
from sklearn.model_selection import train_test_split
# 引入数据分割的工具
from sklearn.preprocessing import StandardScaler
# 引入标准化数据的工具
from sklearn.svm import SVR
# 引入支持向量机回归模型
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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

linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train, Y_train.ravel())
# 二维的调用的时候返回一维的
linear_svr_Y_predict = linear_svr.predict(X_test)

print 'default measurement of linear SVR is', linear_svr.score(X_test, Y_test)
print 'R-squared value of linear SVR is', r2_score(Y_test, linear_svr_Y_predict)
# 由前一节线性回归模型我们可知，r2_score和自带的.score效果是一样的，我们之后用的都是自带的，不用调用r2_score了
print 'the mean squared error of linear SVR is', mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(linear_svr_Y_predict))
# 均方误差
print 'the mean absoluate error of linear SVR is', mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(linear_svr_Y_predict))
# 平均绝对误差

poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train, Y_train.ravel())
poly_svr_Y_predict = poly_svr.predict(X_test)

print 'default measurement of poly SVR is', poly_svr.score(X_test, Y_test)
print 'R-squared value of poly SVR is', r2_score(Y_test, poly_svr_Y_predict)
print 'the mean squared error of poly SVR is', mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(poly_svr_Y_predict))
print 'the mean absoluate error of poly SVR is', mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(poly_svr_Y_predict))

rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train, Y_train.ravel())
rbf_svr_Y_predict = rbf_svr.predict(X_test)
print 'default measurement of rbf SVR is', rbf_svr.score(X_test, Y_test)
print 'R-squared value of rbf SVR is', r2_score(Y_test, rbf_svr_Y_predict)
print 'the mean squared error of rbf SVR is', mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(rbf_svr_Y_predict))
print 'the mean absoluate error of rbf SVR is', mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(rbf_svr_Y_predict))
