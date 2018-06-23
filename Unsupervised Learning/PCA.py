#!/usr/bin/env python    # -*- coding: utf-8 -*

import pandas as pd
# 导入用于数据读取和处理
import numpy as np
# 导入用于数值处理
from sklearn.decomposition import PCA
# 导入PCA用于特征提取
from matplotlib import pyplot as plt
# 导入画图工具包

digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

X_digits = digits_train[np.arange(64)]
Y_digits = digits_train[64]
# 分割训练数据的特征向量和标记

estimator = PCA(n_components=2)
# 初始化一个可以将高维度特征向量压缩为低维度的PCA，设置为2维
X_pca = estimator.fit_transform(X_digits)
# 将X进行PCA压缩处理

def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

    for i in xrange(len(colors)):
        px = X_pca[:, 0][Y_digits.as_matrix() == i]
        py = X_pca[:, 1][Y_digits.as_matrix() == i]
        plt.scatter(px, py, c=colors[i])

    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()

plot_pca_scatter()