#!/usr/bin/env python    # -*- coding: utf-8 -*

"""肘部观察法示例"""
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
# 距离计算函数
import matplotlib.pyplot as plt

# 使用均匀分布函数随机三个簇，每个簇周围10个数据样本
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(5.5, 6.5, (2, 10))
cluster3 = np.random.uniform(3.0, 4.0, (2, 10))

# 绘制30个数据样本的分布图像
X = np.hstack((cluster1, cluster2, cluster3)).T
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# 测试9中不同聚类中心数量下，每种情况的聚类质量
K = range(1, 10)
meandistortions = []

for k in K:
    kmeans = KMeans(n_clusters=k)
    # k个聚类中心,K是循环（1,10）取值的
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    # 聚类分析，计算平均畸变程度

plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average Dispersion')
plt.title('Selecting k with the Elbow Method')
plt.show()