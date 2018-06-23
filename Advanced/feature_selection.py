# -*- coding: utf-8 -*

import numpy as np
import pandas as pd
# 导入pandas工具包
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
# 导入决策树工具包
from sklearn import feature_selection
# 导入特征选择工具包
from sklearn.model_selection import cross_val_score
# 引入交叉验证工具包，和train_test_split一样，是以前的cross_validation留下来的
# 现在都属于model_selection工具包
import pylab as pl
# 导入画图工具包

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

Y = titanic['survived']
X = titanic.drop(['row.names', 'name', 'survived'], axis=1)
# 分离数据特征与预测目标

X['age'].fillna(X['age'].mean(), inplace=True)
X.fillna('UNKNOWN', inplace=True)
# 对缺失数据进行补充

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=33)

vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))
# 类别型数据进行向量化

print len(vec.feature_names_)
# 输出处理后的特征向量的维度

dt = DecisionTreeClassifier(criterion='entropy')
# 决策树分类器的初始化
# criterion='entropy'？？？
dt.fit(X_train, Y_train)
"""对所有特征进行预测，并且评估性能"""
print dt.score(X_test, Y_test)

fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
# 筛选前20%的特征，使用相同配置的决策树模型进行预测
X_train_fs = fs.fit_transform(X_train, Y_train)
"""????这里不清楚为什么fit_transform(中有X和Y_train)"""

dt.fit(X_train_fs, Y_train)
X_test_fs = fs.transform(X_test)

print dt.score(X_test_fs, Y_test)

"""接下来是交叉验证，并作图展示性能随着特征筛选比例的变化"""

percentiles = range(1, 100, 2)
# 设定特征提取的范围1到100，每次加2
results = []
# 结果存为[]列表的形式

for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
    # 特征选择的固定格式
    # feature_selection.SelectPercentile(feature_selection.chi2, percentile=比例)

    X_train_fs = fs.fit_transform(X_train, Y_train)
    scores = cross_val_score(dt, X_train_fs, Y_train, cv=5)
    results = np.append(results, scores.mean())
print results

opt = np.where(results == results.max())[0][0]

# opt = np.where(results == results.max())[0]
# 会报错
# only integer scalar arrays can be converted to a scalar index
# print opt
# 输出[3]

print 'Optimal number of features %d' % percentiles[opt]

"""
print(np.where(results == results.max()))
#返回的是一个(array([3], dtype=int64),)元组形式的数据，我们需要的是这个results.max的索引，3正是索引  
#我们就要想办法把3提取出来，可以看出[3]是一个array
也就是矩阵形式的，那么3所在的位置是一行一列
所以在下一步骤做相应的提取  
opt = np.where(results == results.max())[0][0]
#这一句跟源代码有出入，查看文档np.where返回的是 ndarray or tuple of ndarrays类型数据  
"""

pl.plot(percentiles, results)
# 画出准确率随着特征提取的比例变化图
pl.xlabel('percentiles of feature')
pl.ylabel('accuracy')
pl.show()

fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=7)
# 之前得到的最高的时候为提取7%的特征时

X_train_fs = fs.fit_transform(X_train, Y_train)
dt.fit(X_train_fs, Y_train)
X_test_fs = fs.transform(X_test)
# 检测最高的准确率为多少

print dt.score(X_test_fs, Y_test)