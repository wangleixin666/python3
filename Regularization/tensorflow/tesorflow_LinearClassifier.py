import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train = pd.read_csv('../../data/breast-cancer/breast-cancer-train.csv')
test = pd.read_csv('../../data/breast-cancer/breast-cancer-test.csv')
# print(train.info())
# 读取数据信息

"""分割特征与分类目标"""
X_train = np.float32(train[['Clump Thickness', 'Cell Size']].T)
Y_train = np.float32(train['Type'].T)

X_test = np.float32(test[['Clump Thickness', 'Cell Size']].T)
Y_test = np.float32(test['Type'].T)

b = tf.Variable(tf.zeros([1]))
# 定义一个tensorflow的变量b作为截距，初始值设为1.0
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# 定义一个权重，初始值在-1.0到1.0之间，随机分布，一行两列

y = tf.matmul(W, X_train) + b
# 显示y与x之间的关系

loss = tf.reduce_mean(tf.square(y - Y_train))
# 求均方误差
optimizer = tf.train.GradientDescentOptimizer(0.01)
# 使用梯度下降算法估计参数W，b，设置迭代步长为0.01，与sklearn中SGDRegressor类似

train = optimizer.minimize(loss)
# 求最小的误差

# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
# 初始化所有变量

sess = tf.Session()
sess.run(init)

for step in range(0, 1000):
    sess.run(train)
    if step % 200 == 0:
        print(step, sess.run(W), sess.run(b))
# 每200次输出一次训练结果

test_negative = test.loc[test['Type'] == 0][['Clump Thickness', 'Cell Size']]
test_positive = test.loc[test['Type'] == 1][['Clump Thickness', 'Cell Size']]
# 设置测试样本

plt.scatter(test_negative['Clump Thickness'], test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(test_positive['Clump Thickness'], test_positive['Cell Size'], marker='x', s=150, c='black')
# 画图，恶性的标记为o ，大小为 200，颜色为 红色
# 良性的 标记为 x, 大小为 150 ，颜色为 黑色

plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
# 横纵坐标为 Clump Thickness 和 Cell Size

lx = np.arange(0, 12)
# X的范围为0到12
ly = (0.5 - sess.run(b) - lx * sess.run(W)[0][0]) / sess.run(W)[0][1]

sess.close()

plt.plot(lx, ly, color='green')
# 画图中画一条线
plt.show()
