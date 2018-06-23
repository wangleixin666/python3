import pandas as pd
import numpy as np
# from sklearn import datasets, metrics, preprocessing, model_selection
import skflow
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 电脑CPU占用率高，烧的太快
# from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('../data/MNIST/train.csv')
test = pd.read_csv('../data/MNIST/test.csv')

Y_train = train['label']
X_train = train.drop('label', 1)
# 删除名字为label的那一行/列
X_test = test
"""
使用skflow自带的线性分类器进行预测
classifier = skflow.TensorFlowLinearClassifier(n_classes=10, batch_size=100, steps=1000, learning_rate=0.01)
classifier.fit(X_train, Y_train)

linear_Y_predict = classifier.predict(X_test)
linear_submission = pd.DataFrame({'ImageId':range(1, 28001), 'Label':linear_Y_predict})
linear_submission.to_csv('../data/MNIST/linear_submission.csv', index=False)
"""
"""                    
if len(self.output_shape) == 2:
out.itemset((i, int(self.y[sample])), 1.0)
源码中int(self.y[sample])需要变成整数，不过结果有误差！
"""
"""
使用skflow自带的DNN进行预测
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[200, 50, 10], n_classes=10, steps=5000, learning_rate=0.01, batch_size=50)
classifier.fit(X_train, Y_train)

dnn_Y_predict = classifier.predict(X_test)
dnn_submission = pd.DataFrame({'ImageId': range(1, 28001), 'Label': dnn_Y_predict})
dnn_submission.to_csv('../data/MNIST/dnn_submission.csv', index=False)
"""
"""
使用Tesnsorflow中的算子自行搭建复杂的卷积神经网络，并使用skflow中的程序接口从事MNIST数据的学习和预测
"""


def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_model(X, Y):
    X = tf.reshape(X, [-1, 28, 28, 1])
    with tf.variable_scope('conv_layer1'):
        h_conv1 = skflow.ops.conv2d(X, n_filters=32, filter_shape=[5, 5], bias=True, activation=tf.nn.relu)
        h_pooll = max_pool_2x2(h_conv1)
    with tf.variable_scope('conv_layer2'):
        h_conv2 = skflow.ops.conv2d(h_pooll, n_filters=64, filter_shape=[5, 5], bias=True, activation=tf.nn.relu)
        h_pooll2 = max_pool_2x2(h_conv2)
        h_pooll2_flat = tf.reshape(h_pooll2, [-1, 7 * 7 * 64])
    h_fcl = skflow.ops.dnn(h_pooll2_flat, [1024], activation=tf.nn.relu, keep_prob=0.5)
    return skflow.models.logistic_regression(h_fcl, Y)


classifier = skflow.TensorFlowEstimator(model_fn=conv_model, n_classes=10, batch_size=100, steps=2000, learning_rate=0.001)
classifier.fit(X_train, Y_train)

conv_Y_predict = []
for i in np.arange(100, 28001, 100):
    conv_Y_predict = np.append(conv_Y_predict, classifier.predict(X_test[i-100:i]))
conv_submission = pd.DataFrame({'ImageId': range(1, 28001), 'Label': np.int32(conv_Y_predict)})
conv_submission.to_csv('../data/MNIST/conv_submission.csv',index=False)
