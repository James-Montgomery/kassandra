import random

def shuffle(x, y, N):
    ind_list = [i for i in range(N)]
    random.shuffle(ind_list)
    x = x[ind_list]
    y = y[ind_list]
    return x, y
