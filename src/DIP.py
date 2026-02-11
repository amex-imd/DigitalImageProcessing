# МОДУЛЬ ДЛЯ DIGITAL IMAGE PROCESSING
import numpy as np


def linearTransformation(img, alpha: float = 1.0, beta: float = 0.0):
    res = alpha * img.astype('float64') + beta
    res = np.clip(res, 0, 255).astype('uint8')
    return res

def gammaCorrection(img, gamma: float = 1.0):
    res = np.power(img.astype('float64') / 255, gamma)
    res = np.clip(res * 255, 0, 255).astype('uint8')
    return res

def histogramEqualization(img):
    numbers, _ = np.histogram(img.flatten(), bins=256)
    CDF = numbers.cumsum() # convolution
    totalPixels = img.shape[0] * img.shape[1]
    NormalizedCDF = np.round((CDF - CDF.min()) * 255 / (totalPixels - CDF.min())).astype('uint8')
    return NormalizedCDF[img]
    
