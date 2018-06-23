#!/usr/bin/env python    # -*- coding: utf-8 -*

# import numpy as np
from sklearn.datasets import load_boston
# 加载波士顿房价数据
from sklearn.model_selection import train_test_split
# 用来分割数据
from sklearn.preprocessing import StandardScaler
# 用来标准化数据
from sklearn.linear_model import LinearRegression
# 引入线性回归模型
from sklearn.linear_model import SGDRegressor
# 引入随机梯度回归模型
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

boston = load_boston()
# print boston.DESCR
# 查看数据说明

X = boston.data
Y = boston.target

# print X.shape
# print Y.shape
# X, Y 都是306列，不过，Y只有一行，也就是一维数组,X有13行

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=33)
# 分割数据的训练集和测试集

# print "the max target value is", np.max(boston.target)
# print "the min target value is", np.min(boston.target)
# print "the average target value is", np.mean(boston.target)

ss_X = StandardScaler()
ss_Y = StandardScaler()
# 分别初始化对特征和目标值的标准化器

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

Y_train = ss_Y.fit_transform(Y_train.reshape(-1, 1))
Y_test = ss_Y.transform(Y_test.reshape(-1, 1))
# 分别标准化处理

"""

出现警告：
Traceback (most recent call last):
File "D:/py/Regreesion/LinearRegression.py", line 37, in <module>
Y_train = ss_Y.fit_transform(Y_train)
ValueError: Expected 2D array, got 1D array instead:
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

"""

"""
主要是工具包版本更新造成的，面对上面问题，我们根据上面错误的提示相应的找出出错的两行代码
Y_train = ss_Y.fit_transform(Y_train)
Y_test = ss_Y.transform(Y_test)

y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1,1))
需要把一维数组转化为二维数组形式
"""

"""以下是分别采用线性回归和随机梯度回归对模型进行参数估计以及预测"""
lr = LinearRegression()
lr.fit(X_train, Y_train)
lr_Y_predict = lr.predict(X_test)

sgdr = SGDRegressor(max_iter=5)
# 设置最大跌在次数为5

sgdr.fit(X_train, Y_train.ravel())
sgdr_Y_predict = sgdr.predict(X_test)

"""
我的Y是2D的形式(shapes, 1),要把二维的形式改成1D的形式(shapes, )
这就可以对fit输入的Y_train作y_train.ravel()这样的转换
即把sgdr.fit(Y_train, Y_train)代码修改为sgdr.fit(X_train, Y_train.ravel())
warning就会消失了
"""

"""接下来要评估准确性"""
print 'the value of default measurement of LinearRegression is:', lr.score(X_test, Y_test)

print 'the value of R_squared of LinearRegression is', r2_score(Y_test, lr_Y_predict)

print 'the mean squared error of LinerRegression is', mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(lr_Y_predict))

print 'the mean absolute error of LinerRegression is', mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(lr_Y_predict))

# 接下来是随机梯度的评估
print 'the value of default measurement of SGDRRegression is:', sgdr.score(X_test, Y_test)

print 'the value of R_squared of SGDRRegression is', r2_score(Y_test, sgdr_Y_predict)

print 'the mean squared error of SGDRRegression is', mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(sgdr_Y_predict))

print 'the mean absolute error of SGDRRegression is', mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(sgdr_Y_predict))
