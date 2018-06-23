from sklearn import datasets, metrics, preprocessing, model_selection
import skflow
# import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.ensemble import RandomForestRegressor

boston = datasets.load_boston()
# 加载波士顿房价数据集

x, y = boston.data, boston.target
# 获得数据特征和对应的房价
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=33)
# 分割训练测试集和训练集

"""对数据进行标准化处理"""
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

tf_lr = skflow.TensorFlowLinearRegressor(steps=10000, learning_rate=0.01, batch_size=50)
# 导入skflow中带的tensorflow自带的线性预测模型，设置参数，步数，学习效率
"""
batch_size每次训练的样本数
每次只训练一个样本，即 Batch_Size = 1，这就是在线学习
如果数据集比较小，完全可以采用全数据集 
"""
tf_lr.fit(X_train, Y_train)
tf_lr_Y_predict = tf_lr.predict(X_test)
# 使用该模型训练

"""输出分类器的性能"""
print('the mean absoulute error of tensorflow linear regressor on boston dataset is:', metrics.mean_absolute_error(tf_lr_Y_predict, Y_test))
print('the mean squared error of tensorflow linear regressor on boston dataset is:', metrics.mean_squared_error(tf_lr_Y_predict, Y_test))
print('the mean R-squared error of tensorflow linear regressor on boston dataset is:', metrics.r2_score(tf_lr_Y_predict, Y_test))

tf_dnn_regressor = skflow.TensorFlowDNNRegressor(hidden_units=[100, 40], steps=10000, learning_rate=0.01, batch_size=50)
# 设置每个隐层特征数量的配置
tf_dnn_regressor.fit(X_train, Y_train)
tf_dnn_regressor_Y_predict = tf_dnn_regressor.predict(X_test)

print('the mean absoulute error of tensorflow DNN regressor on boston dataset is:', metrics.mean_absolute_error(tf_dnn_regressor_Y_predict, Y_test))
print('the mean squared error of tensorflow DNN regressor on boston dataset is:', metrics.mean_squared_error(tf_dnn_regressor_Y_predict, Y_test))
print('the mean R-squared error of tensorflow DNN regressor on boston dataset is:', metrics.r2_score(tf_dnn_regressor_Y_predict, Y_test))
# 因为源代码问题，并不能用，源代码中缺少linear函数
# http://blog.csdn.net/sparkexpert/article/details/71513976

"""tf.variable_scope共享变量
def linear(input_, output_size, scope=None):
    '''''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
    '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term
"""

# 在路径中加入linear函数
# C:\Users\WLX\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\contrib\rnn\python\ops\rnn.py中加上

"""但是结果并不准确，只是能运行成功"""

rfr = RandomForestRegressor()
rfr.fit(X_train, Y_train)
rfr_Y_predict = rfr.predict(X_test)
# 导入之前用过的sklearn中随即森林预测进行对比

print('the mean absoulute error of Sklearn RandomForestRegressor on boston dataset is:', metrics.mean_absolute_error(rfr_Y_predict, Y_test))
print('the mean squared error of Sklearn RandomForestRegressor on boston dataset is:', metrics.mean_squared_error(rfr_Y_predict, Y_test))
print('the mean R-squared error of Sklearn RandomForestRegressor on boston dataset is:', metrics.r2_score(rfr_Y_predict, Y_test))
