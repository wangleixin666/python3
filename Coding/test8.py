# 第一题
import sys


def multiple(n):
    flag = 0
    for k in range(n):
        for l in range(k, n, 1):
            if k * l == n:
                if k > l:
                    flag = 1
                    print(k, l)
                elif k < l:
                    flag = 1
                    print(l, k)
                else:
                    pass
    if flag == 0:
        print('No')


if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    value = []
    for i in range(n):
        # 读取每一行
        line = sys.stdin.readline().strip()
        value.append(line)
        # 存入list型的value中
    for j in range(len(value)):
        multiple(int(value[j]))

"""
2
16
9
8 2
No
"""