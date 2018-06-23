#!/usr/bin/env python    # -*- coding: utf-8 -*

import pandas as pd
# 导入pandas用于数据分析
from sklearn.model_selection import train_test_split
# 用来分割数据集
from sklearn.feature_extraction import DictVectorizer
# 用来转换特征向量，特征抽取
from sklearn.tree import DecisionTreeClassifier
# 导入决策树模型
from sklearn.metrics import classification_report
# 导入评估平均准确度，召回率，F1指标

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# 从互联网收集泰坦尼克号乘客数据

# print titanic.head()
# 观察前几行数据

# print titanic.info()
# 使用pandas将数据都转为独有的dataframe格式（二维数据表格），直接使用info查看数据的统计特性

X = titanic[['pclass', 'age', 'sex']]
Y = titanic['survived']
# sex, age, pclass都是决定幸免于否的关键因素

# print X.info()

"""不过数据中age数据列需要补充，sex和pclass需要转化为数值特征0/1代替"""

# 首先补充age里的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略
X['age'].fillna(X['age'].mean(), inplace=True)

"""
出现下列警告：SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame
See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  self._update_inplace(new_data)
"""

# print X.info()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=33)
# 分割数据集

"""凡是类别型的特征都单独剥离出来，独立生成一列特征，数值型的保持不变"""
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
# print vec.feature_names_
X_test = vec.transform(X_test.to_dict(orient='record'))

# 用决策树模型进行训练
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
Y_predict = dtc.predict(X_test)

print dtc.score(X_test, Y_test)
print classification_report(Y_test, Y_predict, target_names=['died', 'survived'])
# 这里的target_names不定，是目标决定的