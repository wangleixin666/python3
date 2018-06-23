"""结果的可视化"""
import tensorflow as tf
import os
import numpy as np
# import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""设置好真实数据的格式"""
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 在-1到1之间取300个点，并且将它格式改为矩阵形式的
# 300行1列，（-0.9, 0），（-0.8, 0）
noise = np.random.normal(0, 0.05, x_data.shape)
# 设置一个噪声，均值是0，方差是0.05，格式和x_data的一致，让y的数据变得更加真实，避免拟合出的结果过好
y_data = np.square(x_data) - 0.5 + noise
# 设置y的格式
# print(y_data) 测试程序的正确性

# 首先设置好传入值，这样就可以每次选择训练的个数，不一定每次都把所有的数据都训练
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    # 300行1列的格式
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

"""tensorflow1.2及以上自带构造新层的函数，不用自己编写add_layer函数了"""
l1 = tf.layers.dense(xs, 10, tf.nn.relu, name='hidden_layer')
output = tf.layers.dense(l1, 1, name='output_layer')
# 并且自己加上name='？'就可以在可视化中显示出来

tf.summary.histogram('h_out', l1)
tf.summary.histogram('pred', output)

with tf.name_scope('loss'):
    loss = tf.losses.mean_squared_error(ys, output, scope='loss')
    # 用这个函数可以代替之前的 tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))

tf.summary.scalar('loss', loss)
# 写进观察变化的Event中

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 对误差进行最小化的训练

init = tf.global_variables_initializer()
# 别忘记对所有变量进行初始化操作
sess = tf.Session()
sess.run(init)
# 先初始化所有的变量

writer = tf.summary.FileWriter('../../data/', sess.graph)
# 把所有的都写入
merge_op = tf.summary.merge_all()
# 一起合并

for steps in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if steps % 50 == 0:
        result = sess.run(merge_op, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, steps)
