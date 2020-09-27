import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time


def constant(arr):
    return 5


def sumOfElements(arr):
    s = 0
    for i in arr:
        s += i

    return s


def mulOfElements(arr):
    s = 0
    for i in arr:
        s *= i

    return s


def polynomCalcus(arr):
    x = 1.5
    s = 0
    for i in range(len(arr)):
        s += arr[i] * math.pow(x, i)
    return s


def BubbleSort(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def QuickSort(arr):
    if len(arr) <= 1:
        return arr
    else:
        q = random.choice(arr)
        s_arr = []
        m_arr = []
        e_arr = []
        for n in arr:
            if n < q:
                s_arr.append(n)
            elif n > q:
                m_arr.append(n)
            else:
                e_arr.append(n)
        return QuickSort(s_arr) + e_arr + QuickSort(m_arr)


def Timsort(arr):
    arr.sort()

def measure_time(n, func):
    arr = np.random.rand(n)
    start = time.time()
    func(arr)
    end = time.time()
    return end - start


def standardMatrixProduct(A, B):
    n = len(A)
    C = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def measureMatrixTime(n):
    arr = np.random.rand(n, n)
    brr = np.random.rand(n, n)
    start = time.time()
    standardMatrixProduct(arr, brr)
    end = time.time()
    return end - start


def draw(x, y, title):
    plt.title(title)
    plt.xlabel('n - number of elements')
    plt.ylabel('execution time')
    plt.plot(x, y)
    plt.grid(True)
    plt.show()


def main():
    matrix_multiply = []
    # const_func = []
    # sum_func = []
    # mul_func = []
    # polynom_func = []
    # bubble_sort_func = []
    # quick_sort_func = []
    # tim_sort_func = []
    #
    ns = [i for i in range(1, 2000)]
    #
    for n in ns:
        print(n)
        # matrix_multiply.append(measureMatrixTime(n))
    #     # const_func.append(measure_time(n, constant))
    #     # sum_func.append(measure_time(n, sumOfElements))
    #     # mul_func.append(measure_time(n, mulOfElements))
    #     # polynom_func.append(measure_time(n, polynomCalcus))
    #     # bubble_sort_func.append(measure_time(n, BubbleSort))
    #     # quick_sort_func.append(measure_time(n, QuickSort))
    #     tim_sort_func.append(measure_time(n, Timsort))
    #
    # # draw(ns, const_func, 'Constant Function')
    # # draw(ns, sum_func, 'Sum Function')
    # # draw(ns, mul_func, 'Mul Function')
    # # draw(ns, polynom_func, 'Polynom Calculating Function')
    # # draw(ns, bubble_sort_func, 'BubbleSort Function')
    # # draw(ns, quick_sort_func, 'QuickSort Function')
    # draw(ns, tim_sort_func, 'TimSort function')
    draw(ns, matrix_multiply, 'Matrix Multiplication')


if __name__ == '__main__':
    main()
