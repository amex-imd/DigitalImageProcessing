# МОДУЛЬ ДЛЯ DIGITAL IMAGE PROCESSING
import numpy as np
import cv2

def linearTransformation(img, alpha: float = 1.0, beta: float = 0.0):
    res = alpha * img.astype('float64') + beta
    res = np.clip(res, 0, 255).astype('uint8')
    return res

def gammaCorrection(img, gamma: float = 1.0):
    res = np.power(img.astype('float64') / 255, gamma)
    res = np.clip(res * 255, 0, 255).astype('uint8')
    return res

def histogramEqualizationSingleChannelImages(img):
    # if len(img.shape) != 1: raise ValueError("TODO")

    nums, _ = np.histogram(img.flatten(), bins=256)
    CDF = nums.cumsum() # convolution
    totalPixels = img.shape[0] * img.shape[1]
    NormalizedCDF = np.round((CDF - CDF.min()) * 255 / (totalPixels - CDF.min())).astype('uint8')
    return NormalizedCDF[img]
    
def histogramEqualizationThreeChannelsImages(img):
    # if len(img.shape) != 3: raise ValueError("TODO")

    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tmp[:, :, 2] = histogramEqualizationSingleChannelImages(tmp[:, :, 2]) # The second channel is brightness

    return cv2.cvtColor(tmp, cv2.COLOR_HSV2BGR)

def integralMatrix(mrx):
    return np.cumsum(np.cumsum(mrx, axis=0), axis=1)

def meanFilterSingleChannelImages(img, kernel_size: int = 3, mode: str = 'edge'):
    gap = kernel_size // 2
    h, w = img.shape

    tmp = np.pad(array=img.astype('float64'), pad_width=2*gap, mode=mode) # 2 * gap to avoid of going beyond borders of image
    integral = integralMatrix(mrx=tmp)
    
    rows1, cols1 = np.ogrid[gap:h+gap, gap:w+gap] # The final image size will be not changed
    rows2, cols2 = rows1+kernel_size, cols1+kernel_size
    S = integral[rows1, cols1] + integral[rows2, cols2] - integral[rows1, cols2] - integral[rows2, cols1]
    res = S / (kernel_size * kernel_size)
    return res.astype('uint8') # ???

def meanFilterThreeChannelsImages(img, kernel_size: int = 3, mode: str = 'edge'):
    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype='uint8') # Using np.empty is better using np.zeroes because it doesn't use full memory
    for i in range(c):
        res[:, :, i] = meanFilterSingleChannelImages(img=img[:, :, i], kernel_size=kernel_size, mode=mode)
    return res

def addSaltAndPepperNoise(img, saltProb: float, pepperProb: float):
    if not 0 <= saltProb <= 1: raise ValueError('The argument \'saltProb\' must be bellow 0 and 1')
    if not 0 <= pepperProb <= 1: raise ValueError('The argument \'pepperProb\' must be bellow 0 and 1')
    if 1 - saltProb - pepperProb < 0: raise ValueError('The arguments \'saltProb\' and \'pepperProb\' are out of range')

    res = np.copy(img)
    randomMask = np.random.uniform(low=0, high=1, size=img.shape[:2])
    saltMask = randomMask < saltProb
    pepperMask = (randomMask >= saltProb) & (randomMask < saltProb + pepperProb)
    if len(img.shape) == 3:
        res[saltMask] = (255, 255, 255)
        res[pepperMask] = (0, 0, 0)
    elif len(img.shape) == 2:
        res[saltMask] = 255
        res[pepperMask] = 0
    return res

def medianFilterSingleChannelImages(img, kernel_size: int = 3, mode: str = 'edge'):
    res = np.empty(shape=img.shape, dtype=img.dtype)
    gap = kernel_size // 2
    h, w = img.shape
    tmp = np.pad(array=img.astype('float64'), pad_width=2*gap, mode=mode) # 2 * gap to avoid of going beyond borders of image
    space = np.lib.stride_tricks.sliding_window_view(tmp, (kernel_size, kernel_size)) # for optimization

    res = np.median(space, axis=(2, 3))
    return np.clip(res[gap:gap+h, gap:gap+w], 0, 255).astype('uint8') # The final image size will be not changed

def medianFilterThreeChannelsImages(img, kernel_size: int = 3, mode: str = 'edge'):
    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype='uint8') # Using np.empty is better using np.zeroes because it doesn't use full memory
    for i in range(c):
        res[:, :, i] = medianFilterSingleChannelImages(img=img[:, :, i], kernel_size=kernel_size, mode=mode)
    return res


def GaussianFilterSingleChannelImages(img, kernel_size: int = 3, sigma: float = 1, mode: str = 'edge'):
    res = np.empty(shape=img.shape, dtype=img.dtype)
    
    gap = kernel_size // 2 # also center
    h, w = img.shape
    tmp = np.pad(array=img.astype('float64'), pad_width=2*gap, mode=mode) # 2 * gap to avoid of going beyond borders of image

    x = np.arange(start=-gap, stop=gap + 1)

    x, y = np.meshgrid(x, x)
    filter = np.exp(np.negative((np.multiply(x, x) + np.multiply(y, y))) / (2 * sigma * sigma))
    filter /= np.sum(filter)
    space = np.lib.stride_tricks.sliding_window_view(tmp, (kernel_size, kernel_size)) # for optimization

    res = np.tensordot(space, filter, axes=((2, 3), (0, 1)))
    return np.clip(res[gap:gap+h, gap:gap+w], 0, 255).astype('uint8') # The final image size will be not changed

def GaussianFilterThreeChannelsImages(img, kernel_size: int = 3, sigma: float = 1, mode: str = 'edge'):
    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype='uint8') # Using np.empty is better using np.zeroes because it doesn't use full memory
    for i in range(c):
        res[:, :, i] = GaussianFilterSingleChannelImages(img=img[:, :, i], kernel_size=kernel_size, sigma=sigma, mode=mode)
    return res

def addGaussianNoise(img, a: float = 0, sigma: float = 1):
    noise = np.random.normal(loc=a, scale=sigma, size=img.shape)
    return np.clip(img + noise, 0, 255).astype('uint8')