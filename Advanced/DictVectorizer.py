# -*- coding: utf-8 -*

from sklearn.feature_extraction import DictVectorizer
# 导入DictVectorizer对特征进行抽取和向量化

measurements = [{'city': 'Dubai', 'temperature': 33.}, {'city': 'London', 'temperature': 12.}, {'city': 'San Fransisco', 'temperature': 18.}]
# 定义一组字典列表来表示数据样本

vec = DictVectorizer()
# 初始化DictVectorizer特征抽取器

print vec.fit_transform(measurements).toarray()
# 输出转化之后的特征矩阵

print vec.get_feature_names()
# 输出各维度特征含义