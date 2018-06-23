import sys

while True:
    try:
        a = sys.stdin.readline().strip()
        if ':' in a:
            i = a.index(':')
            a1 = a[:i]
            a2 = a[i+1:]
            a_1 = a1.split(' ')
            a_2 = a2.split(' ')
            a_1_name = a_1[0]
            a_1_qiu = a_1[1]
            a_2_name = a_2[1]
            a_2_qiu = a_2[0]
            print('A name:', a_1_name)
            print('B name:', a_2_name)
            print('A qiu:', a_1_qiu)
            print('B qiu', a_2_qiu)
    except:
        break