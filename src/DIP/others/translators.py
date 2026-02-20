import numpy as np
import cv2

def linearTransformation(img, alpha: float = 1.0, beta: float = 0.0):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')

    res = alpha * img.astype('float64') + beta
    return np.clip(res, 0, 255).astype('uint8')

def gammaCorrection(img, gamma: float = 1.0):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')

    res = np.power(img.astype('float64') / 255, gamma)
    return np.clip(res * 255, 0, 255).astype('uint8')

def histogramEqualizationSingleChannelImages(img):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if len(img.shape) != 2: raise ValueError('The argument \'img\' must have two dimenshions exactly - \'height\' and \'weight\'')

    nums, _ = np.histogram(img.flatten(), bins=256)
    CDF = nums.cumsum() # convolution
    totalPixels = img.shape[0] * img.shape[1]
    NormalizedCDF = np.round((CDF - CDF.min()) * 255 / (totalPixels - CDF.min())).astype('uint8')
    return NormalizedCDF[img]
    
def histogramEqualizationThreeChannelsImages(img):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 3: raise ValueError('The argument \'img\' must have three dimensions exactly - \'height\', \'weight\' and \'channels\'')
    if len(img.shape) != 3: raise ValueError('The argument \'img\' must have three dimenshions exactly- \'height\', \'weight\' and \'channels\'')

    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tmp[:, :, 2] = histogramEqualizationSingleChannelImages(tmp[:, :, 2]) # The second channel is brightness

    return cv2.cvtColor(tmp, cv2.COLOR_HSV2BGR)

def grayWorld(img):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')

    tmp = img.astype('float64')
    meanR = np.mean(tmp[:, :, 2])
    meanG = np.mean(tmp[:, :, 1])
    meanB = np.mean(tmp[:, :, 0])

    meanVal = (meanR + meanG + meanB) / 3
    tmp[:, :, 2] *= meanVal / (meanR + 1e-12)
    tmp[:, :, 1] *= meanVal / (meanG + 1e-12)
    tmp[:, :, 0] *= meanVal / (meanB + 1e-12)

    return np.clip(tmp, 0, 255).astype('uint8')
