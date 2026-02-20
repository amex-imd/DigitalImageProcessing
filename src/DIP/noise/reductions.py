import numpy as np
import DIP.others.extra_tools as ext
from DIP.others.constants import EPSILON

# Future
from concurrent.futures import ThreadPoolExecutor
from numba import jit, prange

def arithmeticMeanFilterSingleChannelImages(img, filterSize: int = 3, mode: str = 'edge'):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 2: raise ValueError('The argument \'img\' must have two dimensions exactly - \'height\' and \'weight\'')
    if filterSize < 0: raise ValueError('The argument \'filterSize\' must be equal to or greater than 0')
    if filterSize % 2 == 0: raise ValueError('The argument \'filterSize\' must be odd')

    gap = filterSize // 2
    h, w = img.shape

    tmp = np.pad(array=img.astype('float64'), pad_width=gap, mode=mode) # 2 * gap to avoid of going beyond borders of image
    integral = ext.integralMatrix(mrx=tmp)
    
    rows1, cols1 = np.ogrid[0:h, 0:w] # The final image size will be not changed
    rows2, cols2 = rows1+filterSize-1, cols1+filterSize-1
    term1 = np.where((rows1 > 0) & (cols1 > 0), integral[rows1-1, cols1-1], 0.0) 
    term2 = np.where(rows1 > 0, integral[rows1-1, cols2], 0.0)
    term3 = np.where(cols1 > 0, integral[rows2, cols1-1], 0.0)
    term4 = integral[rows2, cols2]
    
    S = term4 - term2 - term3 + term1
    res = S / (filterSize * filterSize)

    return res.astype('uint8')

def arithmeticMeanFilterThreeChannelsImages(img, filterSize: int = 3, mode: str = 'edge'):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 3: raise ValueError('The argument \'img\' must have three dimensions exactly - \'height\', \'weight\' and \'channels\'')
    if filterSize < 0: raise ValueError('The argument \'filterSize\' must be equal to or greater than 0')
    if filterSize % 2 == 0: raise ValueError('The argument \'filterSize\' must be odd')

    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype='uint8') # Using np.empty is better using np.zeroes because it doesn't use memory fully

    for i in range(c): res[:, :, i] = arithmeticMeanFilterSingleChannelImages(img=img[:, :, i], filterSize=filterSize, mode=mode)

    return res

def geometricMeanFilterSingleChannelImages(img, filterSize: int = 3, mode: str = 'edge'):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 2: raise ValueError('The argument \'img\' must have two dimensions exactly - \'height\' and \'weight\'')
    if filterSize < 0: raise ValueError('The argument \'filterSize\' must be equal to or greater than 0')
    if filterSize % 2 == 0: raise ValueError('The argument \'filterSize\' must be odd')

    gap = filterSize // 2
    h, w = img.shape

    tmp = np.pad(array=np.log(img.astype('float64') + EPSILON), pad_width=gap, mode=mode)
    integral = ext.integralMatrix(mrx=tmp)
    
    rows1, cols1 = np.ogrid[0:h, 0:w] # The final image size will be not changed
    rows2, cols2 = rows1+filterSize-1, cols1+filterSize-1
    term1 = np.where((rows1 > 0) & (cols1 > 0), integral[rows1-1, cols1-1], 0.0) 
    term2 = np.where(rows1 > 0, integral[rows1-1, cols2], 0.0)
    term3 = np.where(cols1 > 0, integral[rows2, cols1-1], 0.0)
    term4 = integral[rows2, cols2]
    
    S = term4 - term2 - term3 + term1
    res = np.exp(S / (filterSize * filterSize))

    return np.clip(res, 0, 255).astype('uint8')

def geometricMeanFilterThreeChannelsImages(img, filterSize: int = 3, mode: str = 'edge'):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 3: raise ValueError('The argument \'img\' must have three dimensions exactly - \'height\', \'weight\' and \'channels\'')
    if filterSize < 0: raise ValueError('The argument \'filterSize\' must be equal to or greater than 0')
    if filterSize % 2 == 0: raise ValueError('The argument \'filterSize\' must be odd')

    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype='uint8') # Using np.empty is better using np.zeroes because it doesn't use memory fully

    for i in range(c): res[:, :, i] = geometricMeanFilterSingleChannelImages(img=img[:, :, i], filterSize=filterSize, mode=mode)

    return res

def medianFilterSingleChannelImages(img, filterSize: int = 3, mode: str = 'edge'):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 2: raise ValueError('The argument \'img\' must have two dimensions exactly - \'height\' and \'weight\'')
    if filterSize < 0: raise ValueError('The argument \'filterSize\' must be equal to or greater than 0')
    if filterSize % 2 == 0: raise ValueError('The argument \'filterSize\' must be odd')

    gap = filterSize // 2
    tmp = np.pad(array=img.astype('float64'), pad_width=gap, mode=mode)

    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(filterSize, filterSize))
    res = np.median(space, axis=(2, 3))

    return np.clip(res, 0, 255).astype('uint8') # The final image size will be not changed

def medianFilterThreeChannelsImages(img, filterSize: int = 3, mode: str = 'edge'):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 3: raise ValueError('The argument \'img\' must have three dimensions exactly - \'height\', \'weight\' and \'channels\'')
    if filterSize < 0: raise ValueError('The argument \'filterSize\' must be equal to or greater than 0')
    if filterSize % 2 == 0: raise ValueError('The argument \'filterSize\' must be odd')

    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype='uint8') # Using np.empty is better using np.zeroes because it doesn't use memory fully

    for i in range(c): res[:, :, i] = medianFilterSingleChannelImages(img=img[:, :, i], filterSize=filterSize, mode=mode)

    return res


def midpointFilterSingleChannelImages(img, filterSize: int = 3, mode: str = 'edge'):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 2: raise ValueError('The argument \'img\' must have two dimensions exactly - \'height\' and \'weight\'')
    if filterSize < 0: raise ValueError('The argument \'filterSize\' must be equal to or greater than 0')
    if filterSize % 2 == 0: raise ValueError('The argument \'filterSize\' must be odd')

    gap = filterSize // 2
    tmp = np.pad(array=img.astype('float64'), pad_width=gap, mode=mode)

    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(filterSize, filterSize))
    maxVals = np.max(space, axis=(2, 3))
    minVals = np.min(space, axis=(2, 3))
    res = (minVals + maxVals) / 2

    return res.astype('uint8')

def midpointFilterThreeChannelsImages(img, filterSize: int = 3, mode: str = 'edge'):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 3: raise ValueError('The argument \'img\' must have three dimensions exactly - \'height\', \'weight\' and \'channels\'')
    if filterSize < 0: raise ValueError('The argument \'filterSize\' must be equal to or greater than 0')
    if filterSize % 2 == 0: raise ValueError('The argument \'filterSize\' must be odd')

    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype='uint8') # Using np.empty is better using np.zeroes because it doesn't use memory fully

    for i in range(c): res[:, :, i] = midpointFilterSingleChannelImages(img=img[:, :, i], filterSize=filterSize, mode=mode)
    
    return res

def GaussianFilterSingleChannelImages(img, filterSize: int = 3, sigma: float = 1, mode: str = 'edge'):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 2: raise ValueError('The argument \'img\' must have two dimensions exactly - \'height\' and \'weight\'')
    if filterSize < 0: raise ValueError('The argument \'filterSize\' must be equal to or greater than 0')
    if filterSize % 2 == 0: raise ValueError('The argument \'filterSize\' must be odd')

    res = np.empty(shape=img.shape, dtype=img.dtype)
    
    gap = filterSize // 2 # also center
    tmp = np.pad(array=img.astype('float64'), pad_width=gap, mode=mode)
    x = np.arange(start=-gap, stop=gap + 1)

    x, y = np.meshgrid(x, x)
    filter = np.exp(np.negative((np.multiply(x, x) + np.multiply(y, y))) / (2 * sigma * sigma))
    filter /= np.sum(filter)
    space = np.lib.stride_tricks.sliding_window_view(tmp, (filterSize, filterSize))

    res = np.tensordot(space, filter, axes=((2, 3), (0, 1)))
    return np.clip(res, 0, 255).astype('uint8') # The final image size will be not changed

def GaussianFilterThreeChannelsImages(img, filterSize: int = 3, sigma: float = 1, mode: str = 'edge'):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 3: raise ValueError('The argument \'img\' must have three dimensions exactly - \'height\', \'weight\' and \'channels\'')
    if filterSize < 0: raise ValueError('The argument \'filterSize\' must be equal to or greater than 0')
    if filterSize % 2 == 0: raise ValueError('The argument \'filterSize\' must be odd')

    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype='uint8') # Using np.empty is better using np.zeroes because it doesn't use memory fully

    for i in range(c): res[:, :, i] = GaussianFilterSingleChannelImages(img=img[:, :, i], filterSize=filterSize, sigma=sigma, mode=mode)

    return res

def meanFrames(imgLst):
    if len(imgLst) == 0: raise ValueError('The argument \'imgLst\' must be not empty')

    tmp = [x.astype('float64') for x in imgLst]

    S = np.empty(shape=imgLst[0].shape)
    for x in tmp: S += x

    res = S / len(imgLst)

    return np.clip(res, 0, 255).astype('uint8')