import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(1)
np.random.seed(1)

# 初始数据
n_data = np.ones((100, 2))
# 生成新数组(100, 2)，100行2列，数据类型为默认的numpy.float64 [1., 2.]
x0 = np.random.normal(2*n_data, 1)      # shape (100, 2)
# 均值为2*n_data，方差为1
y0 = np.zeros(100)                      # shape (1, 100)
# 生成1行100列的0 # shape(1, 100)
x1 = np.random.normal(-2*n_data, 1)     # shape (100, 2)
y1 = np.ones(100)                       # shape (1, 100)
# 与zeros相反，生成的数都是1，shape(1,100)
x = np.vstack((x0, x1))                 # shape (200, 2)
# 分行组合到一起，比如[1,2,3]和[4,5,6]变成了[[1,2,3],[4,5,6]]
# shape是(1,3)和(1,3)变成(2,3),而(3,1)和(3,1)变成(6,1)
# 行相加
y = np.hstack((y0, y1))                 # shape (1, 200)
# 直接组合在一起，不分行比如[1,2,3]和[4,5,6]变成了[1,2,3,4,5,6]
# shape是(1,3)和(1,3)变成(1,6)，而(3,1)和(3,1)变成(3,2)
# 列相加

# plot data
plt.scatter(x[:, 0], x[:, 1], c=y, s=100, lw=0, cmap='RdYlGn')
# x[:, 0], x[:, 1]指输入数据的形状shape()数组，c是颜色，s是数组行数，lw为宽度，cmap是颜色
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)     # input x
tf_y = tf.placeholder(tf.int32, y.shape)     # input y

# neural network layers
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)          # hidden layer
output = tf.layers.dense(l1, 2)                     # output layer

loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)           # compute cost
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.squeeze(tf_y), predictions=tf.argmax(output, axis=1),)[1]
# 与回归相比多了一个准确度
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                                                 # control training and others
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# 初始化全局变量，同时初始化局部变量
sess.run(init_op)     # initialize var in graph

plt.ion()   # something about plotting
for step in range(100):
    # train and net output
    _, acc, pred = sess.run([train_op, accuracy, output], {tf_x: x, tf_y: y})
    if step % 50 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x[:, 0], x[:, 1], c=pred.argmax(1), s=100, lw=0, cmap='RdYlGn')
        # c不是普通的颜色，要用pred预测结果的argmax
        # argmax()方法,当调用此方法的函数取最大值(1)时对应的索引的值
        # 不能用某个单一的值，要显示变化
        plt.text(1.5, -4, 'Accuracy=%.2f' % acc, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
# 一定要有ion()和ioff()否则无法显示图
plt.show()
