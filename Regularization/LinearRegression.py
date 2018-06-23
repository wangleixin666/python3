# -*- coding: utf-8 -*

from sklearn.linear_model import LinearRegression
# 导入线性恢复模型
import numpy as np
import matplotlib.pyplot as plt
# 导入画图的工具包

X_train = [[6], [8], [10], [14], [18]]
Y_train = [[7], [9], [13], [17.5], [18]]
# 输入训练样本特征以及目标值

regressor = LinearRegression()
# 初始化线性回归模型
regressor.fit(X_train, Y_train)
# 直接用比萨直径作为特征训练模型

xx = np.linspace(0, 26, 100)
# 在X轴上从0到25军训采样100个数据点
xx = xx.reshape(xx.shape[0], 1)
# 将数据格式转化为二维的
yy = regressor.predict(xx)
# 对xx对应的点进行y值的预测
# 基于以上100个数据点预测回归直线

# 接下来对回归预测的直线进行画图
plt.scatter(X_train, Y_train)
# ？？？

plt1, = plt.plot(xx, yy, label="Degree=1")
# 在图像上标注Degree=1，以xx和yy分别为XY轴作图

plt.axis([0, 25, 0, 25])
# X，Y轴范围均为[0, 25]

plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
# 标注X，Y轴代表的

plt.legend(handles=[plt1])
# ？？？
plt.show()

print 'the R-squared value of Linear Regressor performing on the traning data is:', regressor.score(X_train, Y_train)
# 输出该模型在训练集上的表现