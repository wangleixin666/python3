# 第一题
import sys


def multiple3(n):
    flag = 0
    for k in range(2, int(n/2), 1):

        yu = n % k
        shang = n / k

        if yu == 0:
            flag = 1
            if k > shang:
                flag = 1
                print(k, int(shang))
                break
            else:
                print(int(shang), k)
                break
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
        multiple3(int(value[j]))
