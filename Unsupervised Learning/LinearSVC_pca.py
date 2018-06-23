#!/usr/bin/env python    # -*- coding: utf-8 -*

import pandas as pd
# 导入用于数据读取和处理
import numpy as np
# 导入用于数值处理
from sklearn.decomposition import PCA
# 导入PCA用于特征提取
from sklearn.svm import LinearSVC
# 导入基于线性核的支持向量机分类器
from sklearn.metrics import classification_report

digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

X_train = digits_train[np.arange(64)]
Y_train = digits_train[64]
# 分割训练数据的特征向量和标记

X_test = digits_test[np.arange(64)]
Y_test = digits_test[64]

svc = LinearSVC()
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)
# 使用默认配置初始化LinearSVC,对原始64维特征的训练数据进行建模，并在测试数据进行预测

estimator = PCA(n_components=20)
# 将64维图像数据压缩到20个维度

pca_X_train = estimator.fit_transform(X_train)
# 转化原始训练特征
pca_X_test = estimator.transform(X_test)
# 测试特征按照20个正交维度方向进行转化

pca_svc = LinearSVC()
pca_svc.fit(pca_X_train, Y_train)
pca_Y_predict = pca_svc.predict(pca_X_test)
# 使用默认初始化的LinearSVC对压缩后的20维特征的训练数据进行建模，并作出预测

print svc.score(X_test, Y_test)
print classification_report(Y_test, Y_predict, target_names=np.arange(10).astype(str))
# 对原始图像高维像素特征训练的支持向量机分类器性能作出评估

print pca_svc.score(pca_X_test, Y_test)
print classification_report(Y_test, pca_Y_predict, target_names=np.arange(10).astype(str))
# 对使用PCA压缩后的低维图像特征训练的支持向量机分类器性能作出评估