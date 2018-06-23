#!/usr/bin/env python    # -*- coding: utf-8 -*

from sklearn.datasets import load_digits
# 从sklearn_datasets导入手写数字加载器
from sklearn.model_selection import train_test_split
# split用于分割数据集
from sklearn.preprocessing import StandardScaler
# 导入数据标准化模块
from sklearn.svm import LinearSVC
# 导入基于线性假设的支持向量机svm的线性分类器LinearSVC
from sklearn.metrics import classification_report
# 导入该模块对结果做更加详细的分析

digits = load_digits()
# 从通过数据加载器获得手写体数字的数码图像数据并存储在digits变量中

# print digits.data.shape
# 检视数据规模和特征维度

X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
# 选取75%作为训练,25%作为测试

# print Y_train.shape
# print Y_test.shape
# 检视训练与测试数据规模

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
# 标准化数据

lsvc = LinearSVC()
# 初始化支持向量机分类器LinearSVC

lsvc.fit(X_train, Y_train)
# 进行模型化训练

Y_predict = lsvc.predict(X_test)
# 利用训练好的模型对测试样本的数字类别进行预测

print 'Accuracy of Linear SVC is', lsvc.score(X_test, Y_test)
# 使用模型自带的评估函数进行准确性评测
print classification_report(Y_test, Y_predict, target_names=digits.target_names.astype(str))
# 准确评测分析结果