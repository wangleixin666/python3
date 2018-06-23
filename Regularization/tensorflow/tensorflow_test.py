import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 应该是为了加快CPU运算能力，不然会警告CPU利用不足

import tensorflow as tf
# import numpy as np

greeting = tf.constant('hello world！')

sess = tf.Session()
result = sess.run(greeting)

print(result.decode())
# 应该是编码问题

sess.close()
# 关闭sess
