import numpy as np
import math
import random
import time
import lab21

eps = 0.001

alpha = random.random()
beta = random.random()

normal = np.random.normal(0, 1, 100)
xs = [k / 100 for k in range(0, 100)]
ys = [alpha * xs[k] + beta + normal[k] for k in range(0, 100)]


def f1(x, a, b):
    return a * x + b

def f2(x, a, b):
    return a / (1 + b * x)

def d(f, a, b):
    sum = 0
    for i in range(0, 100):
        sum += (f(xs[i], a, b) - ys[i]) ** 2
    return sum

def exhaustive_search(f):
    a = 0
    b = 10
    minimum = 10000000
    i = a
    j = a
    min_i = 0
    min_j = 0
    iterations = 0
    while i < b:
        while j < b:
            # print(i, j)
            j += 0.01
            val = d(f, i, j)
            if val < minimum:
                minimum = val
                min_i = i
                min_j = j
        i += 0.01
        j = 0

    iterations += 1
    return min_i, min_j


def gauss_search(f):
    a0 = 0
    b0 = 0

    for i in range(2000):
        old_a = a0
        old_b = b0
        if i % 2 == 0:
            new_f = lambda a: d(f, a, b0)
            newa, iter = lab21.exhaustive_search(new_f, 0, 1)
            a0 = newa
        else:
            new_f = lambda b: d(f, a0, b)
            newb, iter = lab21.exhaustive_search(new_f, 0, 1)
            b0 = newb
        if math.fabs(a0 - old_a) < eps and math.fabs(b0 - old_b) < eps:
            return a0, b0

    return a0, b0


def measure_time(title, f, method):
    start = time.time()
    a, b = method(f)
    end = time.time()
    print (a, b, end - start)
    # print(title, ": \n\tmin_a =", a, ", \n\tmin_b = ", b, ",\n\ttime of calculating = ", end - start)


def main():
    measure_time('F1 with gauss', f2, exhaustive_search)


if __name__ == '__main__':
    main()
