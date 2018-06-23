# -*- coding: utf-8 -*

import numpy as np
from sklearn.datasets import fetch_20newsgroups
# 导入文本抓取器
from sklearn.model_selection import train_test_split
# 导入分割工具包
from sklearn.svm import SVC
# 导入支持向量机分类器
from sklearn.feature_extraction.text import TfidfVectorizer
# 导入特征提取工具包
from sklearn.pipeline import Pipeline
# 导入Pipeline，可以简化搭建流程
from sklearn.model_selection import GridSearchCV
# 导入网格搜索模块

news = fetch_20newsgroups(subset='all')
# 从互联网上下载所有数据并存在news中

X_train, X_test, Y_train, Y_test = train_test_split(news.data[:3000], news.target[:3000], test_size=0.25, random_state=33)
# 对前3000条数据进行数据分割

clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer='word')), ('svc', SVC())])
# 将文本抽取与分类器模型串联起来

parameters = {'svc__gamma': np.logspace(-2, 1, 4), 'svc__C': np.logspace(-1, 1, 3)}
# 需要试验的2个超参数的个数分别4,3 svc_gamma的参数共有10^-2， 10^-1.........

"""利用整个CPU的话要加上这句话保护主要循环"""
if __name__ == '__main__':
    gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3, n_jobs=-1)
    # 唯一区别就是n_jobs=-1代表着利用整个CPU
    time_ = gs.fit(X_train, Y_train)
    gs.best_params_, gs.best_score_
    print gs.score(X_test, Y_test)

