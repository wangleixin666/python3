import numpy as np
import matplotlib.pyplot as plt

# 初始数据
n_data = np.ones((5, 2))
# 生成新数组(100, 2)，100行2列，数据类型为默认的numpy.float64 [1., 2.]
x0 = np.random.normal(2*n_data, 1)      # shape (5, 2)
# 均值为2*n_data，方差为1
y0 = np.zeros(5)                      # shape (1, 5)
# 生成1行100列的0 # shape(1, 100)
x1 = np.random.normal(-2*n_data, 1)     # shape (5, 2)
y1 = np.ones(5)                       # shape (1, 5)
# 与zeros相反，生成的数都是1，shape(1,100)
x = np.vstack((x0, x1))                 # shape (10, 2)
# 分行组合到一起，比如[1,2,3]和[4,5,6]变成了[[1,2,3],[4,5,6]]
# shape是(1,3)和(1,3)变成(2,3),而(3,1)和(3,1)变成(6,1)
# 行相加
y = np.hstack((y0, y1))                 # shape (1, 10)
# 直接组合在一起，不分行比如[1,2,3]和[4,5,6]变成了[1,2,3,4,5,6]
# shape是(1,3)和(1,3)变成(1,6)，而(3,1)和(3,1)变成(3,2)
# 列相加
print(x)
plt.scatter(x[:, 0], x[:, 1], c=y, s=100, lw=3, cmap='RdYlGn')
plt.show()
