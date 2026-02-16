import numpy as np
from math import comb

def integralMatrix(mrx):
    return np.cumsum(np.cumsum(mrx, axis=0), axis=1)

def LaplaceKernelWithDiagonals(kernelSize: int = 3):
    n: int = kernelSize // 2
    res = np.empty((kernelSize, kernelSize))
    for i in range(kernelSize):
        for j in range(kernelSize):
            if i == n and j == n: continue
            di, dj = abs(i - n), abs(j - n)
            res[i, j] = comb(n, di) * comb(n, dj)  # From the Internet
    res[n, n] = -np.sum(res)
    return res

def LaplaceKernelWithoutDiagonals(kernelSize: int = 3):
    n: int = kernelSize // 2
    res = np.empty((kernelSize, kernelSize))
    for i in range(kernelSize):
        for j in range(kernelSize):
            if i == n and j == n: continue
            if i == j or i == kernelSize  - j - 1:
                res[i, j] = 0
            else:
                di, dj = abs(i - n), abs(j - n)
                res[i, j] = comb(n, di) * comb(n, dj) # From the Internet
    res[n, n] = -np.sum(res)
    return res
