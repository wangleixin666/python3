import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(1)
np.random.seed(1)

# Hyper parameters
N_SAMPLES = 20
N_HIDDEN = 300
LR = 0.01
# 直接在此处定义好了训练模型的几个参数
# x数据的取值个数，隐藏层的层数，优化器的学习效率

# training data
x = np.linspace(-1, 1, N_SAMPLES)[:, np.newaxis]
y = x + 0.3*np.random.randn(N_SAMPLES)[:, np.newaxis]
# 从标准正态分布中返回一个或多个样本值
# numpy.random.rand(d0, d1, …, dn)的随机样本位于[0, 1)中

# test data
test_x = x.copy()
# 深copy新建一个对象重新分配内存地址，复制对象内容。浅copy不重新分配内存地址，内容指向之前的内存地址
test_y = test_x + 0.3*np.random.randn(N_SAMPLES)[:, np.newaxis]

# show data
plt.scatter(x, y, c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(test_x, test_y, c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
# plt.legend显示图例，loc代表位置,upper left代表在左上方
plt.ylim((-2.5, 2.5))
# 画图时y轴的限制ylim（（范围））
plt.show()

# tf placeholders
tf_x = tf.placeholder(tf.float32, [None, 1])
tf_y = tf.placeholder(tf.float32, [None, 1])
tf_is_training = tf.placeholder(tf.bool, None)
# tf.bool返回True 或者是 False
# 为了在训练和测试的时候控制dropout的率

# overfitting net
o1 = tf.layers.dense(tf_x, N_HIDDEN, tf.nn.relu)
o2 = tf.layers.dense(o1, N_HIDDEN, tf.nn.relu)
# 隐藏层300层，很大几率会产生过拟合问题
o_out = tf.layers.dense(o2, 1)
o_loss = tf.losses.mean_squared_error(tf_y, o_out)
o_train = tf.train.AdamOptimizer(LR).minimize(o_loss)

"""
可以把dropout理解为 模型平均
Dropout的思想是训练整体，并平均整个集合的结果，而不是训练单个
以概率P舍弃部分神经元，其它神经元以概率q=1-p被保留，舍去的神经元的输出都被设置为零
"""
# dropout net
d1 = tf.layers.dense(tf_x, N_HIDDEN, tf.nn.relu)
d1 = tf.layers.dropout(d1, rate=0.5, training=tf_is_training)   # drop out 50% of inputs
"""
def dropout(inputs,
            keep_prob=0.5,
            noise_shape=None,
            is_training=True,
            outputs_collections=None,
            scope=None,
            seed=None):
return outputs = layer.apply(inputs, training=is_training)
apply

apply(
    inputs,
    *args,
    **kwargs
)
Apply the layer on a input.

This simply wraps self.__call__.

Arguments:

inputs: Input tensor(s).
*args: additional positional arguments to be passed to self.call.
**kwargs: additional keyword arguments to be passed to self.call.
Returns:

Output tensor(s).
"""
d2 = tf.layers.dense(d1, N_HIDDEN, tf.nn.relu)
d2 = tf.layers.dropout(d2, rate=0.5, training=tf_is_training)   # drop out 50% of inputs
d_out = tf.layers.dense(d2, 1)
d_loss = tf.losses.mean_squared_error(tf_y, d_out)
d_train = tf.train.AdamOptimizer(LR).minimize(d_loss)
# 换了个优化器，更高阶的优化器，比GradientDescentOptimizer优化器更高级一点

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()   # something about plotting

for t in range(500):
    sess.run([o_train, d_train], {tf_x: x, tf_y: y, tf_is_training: True})  # train, set is_training=True

    if t % 100 == 0:
        # plotting
        plt.cla()
        o_loss_, d_loss_, o_out_, d_out_ = sess.run(
            [o_loss, d_loss, o_out, d_out], {tf_x: test_x, tf_y: test_y, tf_is_training: False}
            # test, set is_training=False
        )
        plt.scatter(x, y, c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x, test_y, c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(test_x, o_out_, 'r-', lw=3, label='overfitting')
        plt.plot(test_x, d_out_, 'b--', lw=3, label='dropout(50%)')
        # label就是图标,plt.legend()可以显示出来
        plt.text(0, -1.2, 'overfitting loss=%.4f' % o_loss_, fontdict={'size': 20, 'color':  'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % d_loss_, fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left')
        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)

plt.ioff()
plt.show()

"""dropout是为了避免过拟合问题的，不同的丢失率会产生不同的结果"""
