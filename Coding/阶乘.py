# 求n的阶乘
# 动态规划算法实现


def jiecheng(n):
    a = []
    a.append(1)
    a.append(1)
    for i in range(2, n+1):
        a.append(i * a[i-1])
    return a[n]


if __name__ == '__main__':
    n = int(input())
    print(jiecheng(n))
