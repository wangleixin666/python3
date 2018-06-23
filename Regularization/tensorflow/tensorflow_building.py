import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def add_layer(inputs, in_size, out_size, avtivation_function=None):
    # 先定义神经网络中增加一层的函数
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 格式是矩阵，In_size * out_size
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 初始值设置为0.1，不推荐设置为0，格式就是1行out_size列

    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # 设置格式Wx+b

    if avtivation_function is None:
        # 并不是一个值，所以不能用 == 只能用is，就好像判断True 和 False一样
        outputs = Wx_plus_b
    else:
        outputs = avtivation_function(Wx_plus_b)
    return outputs


"""设置好真实数据的格式"""
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 在-1到1之间取300个点，并且将它格式改为矩阵形式的
# 300行1列，（-0.9, 0），（-0.8, 0）
noise = np.random.normal(0, 0.05, x_data.shape)
# 设置一个噪声，均值是0，方差是0.05，格式和x_data的一致，让y的数据变得更加真实，避免拟合出的结果过好
y_data = np.square(x_data) - 0.5 + noise
# 设置y的格式
# print(y_data) 测试程序的正确性

"""
建立神经网络（输入层，隐藏层，输出层）
输入层与输出层的格式都是1维的，隐藏层的神经元假设为10个的
"""
# 首先设置好传入值，这样就可以每次选择训练的个数，不一定每次都把所有的数据都训练
xs = tf.placeholder(tf.float32, [None, 1])
# 300行1列的格式
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, avtivation_function=tf.nn.relu)
# 添加隐藏层，输入为1个神经元，输出10个，激励函数设置为最常用的tf.nn.relu函数\\
prediction = add_layer(l1, 10, 1, avtivation_function=None)
# 输出层的输出，激励函数用默认的None，也就是线性关系的，将隐藏层的输出作为输出层的输入

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
# tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]）
# 后半部分不能省略，是函数的处理维度，指tensor向量按照哪些维度求和
# 求真实值与测试值之间的误差
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 对误差进行最小化的训练

init = tf.global_variables_initializer()
# 别忘记对所有变量进行初始化操作

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
# 按照点的形式画出来scatter 而不是连续的

# plt.ion()
# 显示一个图之后继续，不会停止画图
# plt.show()
# 但是并不能用！！！
# plt.show(block=False) # 老版的python

sess = tf.Session()
sess.run(init)
# 先初始化所有的变量
"""
for steps in range(1000):
    sess.run(train_step)
    if steps % 100 == 0:
        print(sess.run(loss))
"""
# 注意传入值还没有赋值的，所有这样是错误的
for steps in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if steps % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        """
        try:
            ax.lines.remove(lines[0])
            # 第一次抹除的话因为未定义，会报错，所以用try
        except Exception:
            pass
        """
        # plt.cla() # 应该是抹除画的图像
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        # plt.pause(0.1)
        # 暂停0.1秒
        # plt.ion()
        # plt.show()
        # 注意缩进的位置，
    # plt.show()
    # 缩进到if结束也就是第一次if后的结果
# plt.ioff()
plt.show()
"""
缩进到头的话，加上try就是for循环结束后，显示最终的曲线
不加try和pause 的话会显示多条曲线
只加pause会显示错误的曲线
如果去掉try，将抹除工作放在画线之后的话，不显示线，因为直接抹除了。。
ax.lines.remove(lines[0])
而如果将pause放在抹除之前的话也是只显示一条错误曲线
plt.pause(0.1)
ax.lines.remove(lines[0])
"""
