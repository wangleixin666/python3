import sys

while True:
    try:
        a = sys.stdin.readline().strip()
        b = int(sys.stdin.readline().strip())
        x = a[:b]
        y = ''.join([i for i in x if not i.isdigit()])
        print(y)
    except:
        break