import numpy as np
import math
import random
import time
import lab21
from scipy.optimize import minimize_scalar, least_squares, leastsq
from scipy.misc import derivative



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

def d1(arr):
    a, b = arr[0], arr[1]
    sum = 0
    for i in range(0, 100):
        sum += (f1(xs[i], a, b) - ys[i]) ** 2
    return sum

def d2(arr):
    a, b = arr[0], arr[1]
    sum = 0
    for i in range(0, 100):
        sum += (f2(xs[i], a, b) - ys[i]) ** 2
    return sum

def dxx1(arr):
    a, b = arr[0], arr[1]
    sum = 0
    for i in range(0, 100):
        sum += 2 * (xs[i] ** 2)
    return sum

def dyy1(arr):
    return 2

def dxx2(arr):
    a, b = arr[0], arr[1]
    sum = 0
    for i in range(0, 100):
        sum += 2 / (b * xs[i] + 1) ** 2
    return sum

def dyy2(arr):
    a, b = arr[0], arr[1]
    sum = 0
    for i in range(0, 100):
        x = xs[i]
        y = ys[i]
        sum += (2 * a * (x ** 2) * (3 * a - 2 * (y + b * x * y)))/((1 + b * x) ** 4)
    return sum

def dxy1(arr):
    a, b = arr[0], arr[1]
    sum = 0
    for i in range(0, 100):
        sum += 2 * xs[i]

    return sum

def dxy2(arr):
    a, b = arr[0], arr[1]
    sum = 0
    for i in range(0, 100):
        x = xs[i]
        y = ys[i]
        sum += - (2 * x * (-2 * a + b * x * y + y)) / ((b * x + 1) ** 3)

    return sum


def grad(d, arr):
    x = arr[0]
    y = arr[1]
    dd1 = lambda a: d([a, y])
    dd2 = lambda b: d([x, b])
    return np.array([derivative(dd1, x), derivative(dd2, y)])

def hessian(which, arr):
    x = arr[0]
    y = arr[1]

    dxx = dxx1([x, y]) if which == 1 else dxx2([x, y])
    dyy = dyy1([x, y]) if which == 1 else dyy2([x, y])
    dxy = dxy1([x, y]) if which == 1 else dxy2([x, y])

    arr1 = np.array([dxx, dxy])
    arr2 = np.array([dxy, dyy])

    return np.array([arr1, arr2])



def minimize(d, a):
    l_min = minimize_scalar(lambda l: d(a - l * grad(d, a))).x
    return l_min

def gradient_decent(d):
    x = np.array([0., 0.])
    old_x = np.array([1., 1.])
    iterations = 0
    while (math.fabs(x[0] - old_x[0]) > eps and math.fabs(x[1] - old_x[1]) > eps) and iterations < 2000:
        old_x = x
        l = minimize(d, x)
        x = x - l * grad(d, x)
        print(iterations)
        iterations += 1
    return x

def conjuregate_gradient_descent(d):
    x = np.array([0., 0.])
    old_x = np.array([1., 1.])
    iterations = 0

    dx = -grad(d, x)
    alpha = minimize_scalar(lambda l: d(x + l * dx)).x
    x = x + dx * alpha
    old_dx = dx
    s = dx
    while (math.fabs(x[0] - old_x[0]) > eps and math.fabs(x[1] - old_x[1]) > eps) and iterations < 2000:
        old_x = x
        dx = -grad(d, x)
        beta = np.dot(dx, np.transpose(dx))/np.dot(old_dx, np.transpose(old_dx))

        s = dx + beta * s
        alpha = minimize_scalar(lambda l: d(x + l * s)).x
        x = x + alpha * s
        print(iterations)
        iterations += 1
    return x

def newton(which, d):
    x = np.array([1., 1.])
    old_x = np.array([0., 0.])
    iterations = 0
    while (math.fabs(x[0] - old_x[0]) > eps and math.fabs(x[1] - old_x[1]) > eps) and iterations < 2000:
        old_x = x
        x = x - np.dot(np.linalg.inv(hessian(which, x)), np.array(grad(d, x)))
        print(iterations)
        iterations += 1

    return x

# def levenberg(d):
#     func = lambda a, b: d([a, b])
#     res = leasts_square(func, np.array([0., 0.]), method='lm')
#     print(res.nfev)
#     return res.x


def main():
    x1 = gradient_decent(d1)
    # x2 = newton(1, d1)
    x3 = conjuregate_gradient_descent(d1)
    # x4 = levenberg(d1)

    print(x1)
    # print(x2)
    print(x3)
    # print(x4)

if __name__ == '__main__':
    main()
