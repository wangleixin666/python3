# -*- coding: utf-8 -*
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# 导入PolynomialFeatures多项式工具包
from sklearn.linear_model import Ridge
# 导入L1范数正则化模型

X_train = [[6], [8], [10], [14], [18]]
Y_train = [[7], [9], [13], [17.5], [18]]
# 输入训练样本特征以及目标值

regressor_poly4 = LinearRegression()
# 初始化四次多项式回归模型
poly4 = PolynomialFeatures(degree=4)
X_train_poly4 = poly4.fit_transform(X_train)
# 使用PolynomialFeatures(degree=4)映射出4次多项式特征，存储在X_train_poly4中
regressor_poly4.fit(X_train_poly4, Y_train)
# 利用4次多项式回归模型进行训练

ridge_poly4 = Ridge()
# 默认配置初始化Lasso模型
ridge_poly4.fit(X_train_poly4, Y_train)
# 使用Lasso对4次多项式特征进行拟合

X_test = [[6], [8], [11], [16]]
Y_test = [[8], [12], [15], [18]]

X_test_poly4 = poly4.transform(X_test)

print regressor_poly4.coef_
# 输出模型的参数列表
print np.sum(regressor_poly4.coef_ ** 2)
# 输出参数的平方和

print ridge_poly4.coef_
# 输出拟合之后恢复模型的参数列表
print np.sum(ridge_poly4.coef_ ** 2)
# 输出拟合后参数的平方和
