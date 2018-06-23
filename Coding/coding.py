def solution(seq, n):
    i = 0
    j = 0
    ret = ''
    while i < len(seq) and j < n:
        if not seq[i].isalpha() and j+2 <= n:
            ret += seq[i:i+3]
            i += 3
            j += 2
        elif seq[i].isalpha():
            # Èç¹ûÊÇ×ÖÄ¸
            ret += seq[i]
            i += 1
            j += 1
        elif j+2 > n:
            break
    result = ''.join([i for i in ret if not i.isdigit()])
    return result


try:
    while True:
        seq = input()
        # print(len(seq.encode('gbk')))
        n = int(input())
        print(solution(seq, n))
except:
    pass
