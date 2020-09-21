import numpy as np
import math
import time

eps = 0.001
phi = (1 + math.sqrt(5)) / 2


def f1(x):
    return x ** 3


def f2(x):
    return math.fabs(x - 0.2)


def f3(x):
    return x * math.sin(1 / x)


def exhaustive_search(f, a, b):
    minimum = 10000000
    i = a
    iterations = 0
    while i < b:
        if f(i) < minimum:
            minimum = f(i)
        i += eps
        iterations += 1

    return minimum, iterations


def dichotomy_search(f, a, b):
    I = 0.01
    iterations = 0
    while b - a > I:
        m1 = (a + b) / 2 - eps
        m2 = (a + b) / 2 + eps
        f1 = f(m1)
        f2 = f(m2)

        if f1 < f2:
            b = m2
        else:
            a = m1
        iterations += 1

    return f((a + b) / 2), iterations


def golden_section_search(f, a, b):
    iterations = 0
    while b - a > eps:
        x1 = b - (b - a) / phi
        x2 = a + (b - a) / phi

        y1 = f(x1)
        y2 = f(x2)
        if (y1 >= y2):
            a = x1
        else:
            b = x2
        iterations += 1

    return f((a + b) / 2), iterations


def measure_time(title, f, method, a=0., b=1):
    start = time.time()
    minimum, iterations = method(f, a, b)
    end = time.time()

    print(title, ": \n\tmin =", minimum, ", \n\titerations = ", iterations, ",\n\ttime of calculating = ", end - start)


def main():
    f = f1
    measure_time("x^3 using exhaustive search", f, exhaustive_search)
    measure_time("x^3 using dichotomy search", f, dichotomy_search)
    measure_time("x^3 using golden section search", f, golden_section_search)
    f = f2
    measure_time("|x - 0.2| using exhaustive search", f, exhaustive_search)
    measure_time("|x - 0.2| using dichotomy search", f, dichotomy_search)
    measure_time("|x - 0.2| using golden section search", f, golden_section_search)
    f = f3
    measure_time("x * sin(1 / x) using exhaustive search", f, exhaustive_search, 0.01)
    measure_time("x * sin(1 / x) using dichotomy search", f, dichotomy_search, 0.01)
    measure_time("x * sin(1 / x) using golden section search", f, golden_section_search, 0.01)


if __name__ == '__main__':
    main()
