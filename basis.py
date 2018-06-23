#!/usr/bin/env python    # -*- coding: utf-8 -*
# 第一行必须要有，后面才能写中文

import pandas as pd   # 导入pandas工具包，用来读取文件，用到了pd.read_csv方法
from matplotlib import pyplot as plt
# 导入pyplot并命名为plt, 属于matplotlib的一个画图的工具 # 不换行会报错？？
import numpy as np    # 导入数据处理的包，用来处理数据,用到了np.random.random([1])，np.arange(0, 12)
from sklearn.linear_model import LogisticRegression   # 引入线性回归分类器

# 调用pandas中的read_csv函数,传入测试文件地址参数，获取返回的数据并且存入df_train和df_test
df_train = pd.read_csv('E:/python/Datasets/Breast-Cancer/breast-cancer-train.csv')
df_test = pd.read_csv('E:/python/Datasets/Breast-Cancer/breast-cancer-test.csv')

# 选取两个特征'Clump Thickness', 'Cell Size'构建测试集中的正负样本,Type为0和1区分样本
df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]

# 绘制良性肿瘤样本点，标记为红色的O，恶性绘制为黑色的X
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')

# 绘制x,y坐标轴说明
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')

# 利用numpy中的random函数随机采样直线的截距和系数,画一条蓝色直线1
intercept = np.random.random([1])
coef = np.random.random([2])
lx = np.arange(0, 12)
ly = (-intercept - lx * coef[0]) / coef[1]

plt.plot(lx, ly, c='blue')

# 使用前10条训练样本学习直线的系数和截距,画一条黄色直线2
lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10], df_train['Type'][:10])
print("使用前10条训练样本的误差:", lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))
intercept = lr.intercept_
coef = lr.coef_[0, :]
ly = (-intercept - lx * coef[0]) / coef[1]

plt.plot(lx, ly, c='yellow')

# 使用所有训练样本学习直线的系数和截距，画一条红色直线3
lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type'])
print("使用所有训练样本的误差:", lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))
intercept = lr.intercept_
coef = lr.coef_[0, :]
ly = (-intercept - lx * coef[0]) / coef[1]

plt.plot(lx, ly, c='red')

plt.show()