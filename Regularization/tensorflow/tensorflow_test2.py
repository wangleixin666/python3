import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# import numpy as np

matrix1 = tf.constant([[3., 3.]])
# 生成一个一行两列的矩阵
matrix2 = tf.constant([[2.], [2.]])
# 生成一个两行一列的矩阵

product = tf.matmul(matrix1, matrix2)
# matmul是指矩阵相乘的函数
# matrix multiple的缩写

linear = tf.add(product, tf.constant(2.0))
# 将矩阵乘积product与一个标量2.0相结合，作为最后的结果

# sess = tf.Session()
# print(sess.run(linear))
# sess.close()
# 这种方法过于麻烦，还要关闭，因此换一种较为简单的调用session的方法

with tf.Session() as sess:
    print(sess.run(linear))