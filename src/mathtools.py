import numpy as np
from math import comb

def integralMatrix(mrx):
    return np.cumsum(np.cumsum(mrx, axis=0), axis=1)

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

def LaplaceFilter(filterSize: int = 3):
    fx, fy = SobelFilter(filterSize=filterSize)
    fxx, fyy = np.pad(array=fx.astype('float64'), pad_width=filterSize//2, mode='constant', constant_values=0), np.pad(array=fy.astype('float64'), pad_width=filterSize//2, mode='constant', constant_values=0)

    sx, sy = np.lib.stride_tricks.sliding_window_view(fxx, (filterSize, filterSize)), np.lib.stride_tricks.sliding_window_view(fyy, (filterSize, filterSize)) # for optimization
    tx, ty = np.tensordot(sx, fx, axes=((2, 3), (0, 1))), np.tensordot(sy, fy, axes=((2, 3), (0, 1)))
    res = tx + ty
    res = res - np.mean(res)
    return res / np.max(np.abs(res))


def RobertsonFilter():
    return np.array([[1, 0], [0, -1]], dtype='float64'), np.array([[0, 1], [-1, 0]], dtype='float64')

