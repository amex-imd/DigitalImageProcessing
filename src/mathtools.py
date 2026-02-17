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
    n = kernelSize // 2
    res = np.zeros((kernelSize, kernelSize))
    
    for i in range(kernelSize):
        for j in range(kernelSize):
            if i == n and j == n:
                continue
            
            di = abs(i - n)
            dj = abs(j - n)
            
            if (di == 0 and dj > 0) or (dj == 0 and di > 0):
                dist = max(di, dj)
                res[i, j] = -1.0 / dist
    
    res[n, n] = -np.sum(res)
    res = res / np.sum(np.abs(res))
    
    return res

def SobelFilter(filterSize: int = 3):
    n = filterSize // 2
    res = np.empty(shape=(filterSize, filterSize))
    binom = np.array([comb(2*n, i) for i in range(2*n+1)])
    
    diff = np.zeros(filterSize)
    for j in range(filterSize):
        if j < n:
            diff[j] = - (n - j) * comb(2*n, j)
        elif j > n:
            diff[j] = (j - n) * comb(2*n, j)
            # TODO diff[j] = 0
    
    # Finding GCD for all no zero values to normalize
    tmp = np.abs(diff[diff != 0])
    if len(tmp) > 0:
        gcd_val = np.gcd.reduce(tmp.astype(int))
        diff = (diff / gcd_val).astype(int)
    
    for i in range(filterSize):
        for j in range(filterSize):
            res[i, j] = binom[i] * diff[j]
    
    return res, np.rot90(res)

def RobertsonFilter():
    return np.array([[1, 0], [0, -1]], dtype='float64'), np.array([[0, 1], [-1, 0]], dtype='float64')

