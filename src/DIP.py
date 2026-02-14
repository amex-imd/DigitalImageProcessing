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
    
    rows1, cols1 = np.ogrid[gap:h+gap, gap:w+gap]
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





