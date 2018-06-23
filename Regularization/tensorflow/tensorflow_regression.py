import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tf.set_random_seed(1)
# np.random.seed(1)
# 作用？# 生成随机张量
# 使所有op产生的随机序列在会话之间是可重复的，设置一个图级别的seed

# 构造数据
x = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise                          # shape (100, 1) + some noise
# 就相当于np.square(x) + noise

# plot data
plt.scatter(x, y)
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)     # input x
tf_y = tf.placeholder(tf.float32, y.shape)     # input y
# 传入值，可以设置每次训练的个数，不一定是每次都训练所有

# 神经网络的构造
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)          # hidden layer隐藏层

output = tf.layers.dense(l1, 1)                     # output layer输出层
# 将隐藏层的输出 作为输入，输出数据个数为1个，激励函数为默认的线性函数

loss = tf.losses.mean_squared_error(tf_y, output)   # compute cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph

plt.ion()   # something about plotting
# 可以在每次plt.show()结束后继续画图不停止

for step in range(100):
    # train and net output
    _, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
    # 就相当于把三个run的过程结合到一起
    # 因为三个都有传入值，前面为存入哪个变量中，train_op不用存入，因此用_代替
    if step % 5 == 0:
        # plot and show learning process
        plt.cla()
        # 应该是抹除现有的图像
        plt.scatter(x, y)
        # 然后再画点
        plt.plot(x, pred, 'r-', lw=5)
        # 画出需要的曲线
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        # 文字标记
        plt.pause(0.1)
        # 每次画图间隔0.1s

plt.ioff()
plt.show()
