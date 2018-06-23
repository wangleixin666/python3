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

poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train)
# 使用PolynomialFeatures(degree=2)映射出2次多项式特征，存储在X_train_poly2中

regressor_poly2 = LinearRegression()
# 初始化二次多项式回归模型
regressor = LinearRegression()
# 初始化线性回归模型

regressor_poly2.fit(X_train_poly2, Y_train)
# 利用2次多项式回归模型进行训练
regressor.fit(X_train, Y_train)
# 直接用比萨直径作为特征训练模型

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

plt.scatter(X_train, Y_train)

plt1, = plt.plot(xx, yy, label='Degree=1')
plt2, = plt.plot(xx, yy_poly2, label='Degree=2')
# X轴就是0到25均匀分割的100个点，Y轴是根据不同的函数形式得到的Y值，保持X轴一样才能画图

plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1, plt2])
plt.show()

print 'the R-squared value of Polynominal Regressor(Degree=2) performing on the training data is:', regressor_poly2.score(X_train_poly2, Y_train)