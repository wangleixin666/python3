def dicesSum(n):
    if n == 0:
        return None
    result = [[1, 1, 1, 1, 1, 1], ]
    for i in range(1,n):
        x = 5*(i+1)+1
        result.append([0 for _ in range(x)])
        for j in range(x):
            if j < 6:
                result[i][j] = (sum(result[i-1][0:j+1]))
            elif 6 <= j <= 3*i+2:
                result[i][j] = (sum(result[i-1][j-5:j+1]))
            else:
                break
        left = 0
        right = len(result[i]) - 1
        while left <= right:
            result[i][right] = result[i][left]
            left += 1
            right -= 1
    res = result[-1]
    all = float(sum(res))
    other = []

    for i, item in enumerate(res):
        pro = item / all
        pro_new = float('%.5f' % pro)
        other.append([n+i, pro_new])
    return other


import sys
n = int(sys.stdin.readline().strip())
print(dicesSum(n))
