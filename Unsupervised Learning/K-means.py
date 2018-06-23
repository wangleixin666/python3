#!/usr/bin/env python    # -*- coding: utf-8 -*

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
# 导入聚类用的KMeans
from sklearn.metrics import silhouette_score
# 导入准确度评估的
import matplotlib.pyplot as plt
# 画图工具包
from sklearn import metrics

digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)
# header参数指定columns都为从0自增长的数
# header = 0表示数据的第一行，而不是文件的第一行
# 如果未传递names，默认行为就好像设置为0，否则None

X_train = digits_train[np.arange(64)]
Y_train = digits_train[64]
# 从训练集和测试集都分理处64维度的像素特征与1维度的数字目标
X_test = digits_test[np.arange(64)]
Y_test = digits_test[64]

kmeans = KMeans(n_clusters=10)
# 初始化KMeans模型，并将初始聚类中心设为10
kmeans.fit(X_train)

Y_pred = kmeans.predict(X_test)
# 逐条判断测试图像所属的聚类中心

print metrics.adjusted_rand_score(Y_test, Y_pred)
# 聚类性能评估

"""分割出3*2个子图，并在1号子图作图"""
plt.subplot(3, 2, 1)

x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
X = np.array(zip(x1, x2)).reshape(len(x1), 2)
# 初始化原始数据点

plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Instances')
plt.scatter(x1, x2)
# 在1号子图做出原始数据点阵的分布

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']
# 将不同类的元素绘制成不同的颜色和标记

clusters = [2, 3, 4, 5, 8]
subplot_counter = 1
sc_scores = []

for t in clusters:
    subplot_counter += 1
    plt.subplot(3, 2, subplot_counter)
    kmeans_model = KMeans(n_clusters=t).fit(X)

    for i, l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls='None')
        plt.xlim([0, 10])
        plt.ylim([0, 10])

    """注意python中的函数缩进问题，不然很容易产生错误"""
    sc_score = silhouette_score(X, kmeans_model.labels_, metric='euclidean')
    sc_scores.append(sc_score)
    plt.title('K=%s, silhouette coefficient=%0.03f' % (t, sc_score))
    # 绘制轮廓系数与不同类 数量的直观显示图

#  绘制关系曲线
plt.figure()
plt.plot(clusters, sc_scores, '*-')
plt.plot(clusters, sc_scores, '*-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient Score')
plt.show()
