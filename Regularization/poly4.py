# -*- coding: utf-8 -*

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# 导入PolynomialFeatures多项式工具包
import matplotlib.pyplot as plt
# 导入画图工具包

X_train = [[6], [8], [10], [14], [18]]
Y_train = [[7], [9], [13], [17.5], [18]]
# 输入训练样本特征以及目标值

regressor = LinearRegression()
# 初始化线性回归模型
regressor.fit(X_train, Y_train)
# 直接用比萨直径作为特征训练模型

regressor_poly2 = LinearRegression()
# 初始化二次多项式回归模型
poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train)
# 使用PolynomialFeatures(degree=2)映射出2次多项式特征，存储在X_train_poly2中
regressor_poly2.fit(X_train_poly2, Y_train)
# 利用2次多项式回归模型进行训练

regressor_poly4 = LinearRegression()
# 初始化四次多项式回归模型
poly4 = PolynomialFeatures(degree=4)
X_train_poly4 = poly4.fit_transform(X_train)
# 使用PolynomialFeatures(degree=4)映射出4次多项式特征，存储在X_train_poly4中
regressor_poly4.fit(X_train_poly4, Y_train)
# 利用4次多项式回归模型进行训练

xx = np.linspace(0, 26, 100)
# 在X轴上从0到25军训采样100个数据点
xx = xx.reshape(xx.shape[0], 1)
# 将数据格式转化为二维的
yy = regressor.predict(xx)
# 对xx对应的点进行y值的预测

xx_poly2 = poly2.transform(xx)
# 用新映射映射出X轴的数据xx_poly2
# 先映射为2次多项式特征
"""
如果不进行转化，数据格式不对
shapes (100,1) and (3,1) not aligned: 1 (dim 1) != 3 (dim 0)
"""
yy_poly2 = regressor_poly2.predict(xx_poly2)
# 预测出Y轴的数据

xx_poly4 = poly4.transform(xx)
yy_poly4 = regressor_poly4.predict(xx_poly4)
# 预测出Y轴的数据

plt.scatter(X_train, Y_train)

plt1, = plt.plot(xx, yy, label='Degree=1')
plt2, = plt.plot(xx, yy_poly2, label='Degree=2')
# X轴就是0到25均匀分割的100个点，Y轴是根据不同的函数形式得到的Y值，保持X轴一样才能画图
plt3, = plt.plot(xx, yy_poly4, label='Degree=4')

plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1, plt2, plt3])
plt.show()

X_test = [[6], [8], [11], [16]]
Y_test = [[8], [12], [15], [18]]

print 'the R-squared value of Polynominal Regressor(Degree=1) performing on the training data is:', regressor.score(X_train, Y_train)
print 'the R-squared value of Polynominal Regressor(Degree=1) performing on the test data is:', regressor.score(X_test, Y_test)

X_test_poly2 = poly2.transform(X_test)
print 'the R-squared value of Polynominal Regressor(Degree=2) performing on the training data is:', regressor_poly2.score(X_train_poly2, Y_train)
print 'the R-squared value of Polynominal Regressor(Degree=2) performing on the test data is:', regressor_poly2.score(X_test_poly2, Y_test)

X_test_poly4 = poly4.transform(X_test)
print 'the R-squared value of Polynominal Regressor(Degree=4) performing on the training data is:', regressor_poly4.score(X_train_poly4, Y_train)
print 'the R-squared value of Polynominal Regressor(Degree=4) performing on the test data is:', regressor_poly4.score(X_test_poly4, Y_test)