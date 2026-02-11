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

    numbers, _ = np.histogram(img.flatten(), bins=256)
    CDF = numbers.cumsum() # convolution
    totalPixels = img.shape[0] * img.shape[1]
    NormalizedCDF = np.round((CDF - CDF.min()) * 255 / (totalPixels - CDF.min())).astype('uint8')
    return NormalizedCDF[img]
    
def histogramEqualizationThreeChannelImages(img):
    # if len(img.shape) != 3: raise ValueError("TODO")

    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tmp[:, :, 2] = histogramEqualizationSingleChannelImages(tmp[:, :, 2]) # The second channel is brightness

    return cv2.cvtColor(tmp, cv2.COLOR_HSV2BGR)

def integralMatrix(mrx):
    return np.cumsum(np.cumsum(mrx, axis=0), axis=1)