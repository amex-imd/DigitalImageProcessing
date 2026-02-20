import numpy as np
import math

def integralMatrix(mrx):
    if mrx.size == 0: raise ValueError('')

    return np.cumsum(np.cumsum(mrx, axis=0), axis=1)

def SobelFilter(filterSize: int = 3):
    if filterSize < 0: raise ValueError('The argument \'filterSize\' must be equal to or greater than 0')
    if filterSize % 2 == 0: raise ValueError('The argument \'filterSize\' must be odd')

    n = filterSize // 2

    binomials = np.array([math.comb(2*n, j) for j in range(filterSize)])
    indeces = np.arange(filterSize)
    diff = np.zeros(filterSize)
    mask = indeces < n
    diff[mask] = -(n - indeces[mask]) * binomials[mask]
    mask = indeces > n
    diff[mask] = (indeces[mask] - n) * binomials[mask]

    Gx = np.outer(binomials, diff)
    Gy = np.rot90(Gx)
    
    return Gx, Gy

def LaplaceFilter(filterSize: int = 3):
    if filterSize < 0: raise ValueError('The argument \'filterSize\' must be equal to or greater than 0')
    if filterSize % 2 == 0: raise ValueError('The argument \'filterSize\' must be odd')
    
    fx, fy = SobelFilter(filterSize=filterSize)
    fxx, fyy = np.pad(array=fx.astype('float64'), pad_width=filterSize//2, mode='constant', constant_values=0), np.pad(array=fy.astype('float64'), pad_width=filterSize//2, mode='constant', constant_values=0)

    sx, sy = np.lib.stride_tricks.sliding_window_view(fxx, (filterSize, filterSize)), np.lib.stride_tricks.sliding_window_view(fyy, (filterSize, filterSize))
    tx, ty = np.tensordot(sx, fx, axes=((2, 3), (0, 1))), np.tensordot(sy, fy, axes=((2, 3), (0, 1)))
    res = tx + ty
    res = res - np.mean(res)

    return res / np.max(np.abs(res))

def RobertsonFilter():
    return np.array([[1, 0], [0, -1]], dtype='float64'), np.array([[0, 1], [-1, 0]], dtype='float64')