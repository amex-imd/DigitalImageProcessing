import numpy as np
import cv2
from constants import EPSILON

def linearTransformation(img, alpha: float = 1.0, beta: float = 0.0):
    res = alpha * img.astype('float64') + beta
    return np.clip(res, 0, 255).astype('uint8')

def gammaCorrection(img, gamma: float = 1.0):
    res = np.power(img.astype('float64') / 255, gamma)
    return np.clip(res * 255, 0, 255).astype('uint8')

def histogramEqualizationSingleChannelImages(img):
    nums, _ = np.histogram(img.flatten(), bins=256)
    CDF = nums.cumsum() # convolution
    totalPixels = img.shape[0] * img.shape[1]
    NormalizedCDF = np.round((CDF - CDF.min()) * 255 / (totalPixels - CDF.min())).astype('uint8')
    return NormalizedCDF[img]
    
def histogramEqualizationThreeChannelsImages(img):
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tmp[:, :, 2] = histogramEqualizationSingleChannelImages(tmp[:, :, 2]) # The second channel is brightness

    return cv2.cvtColor(tmp, cv2.COLOR_HSV2BGR)

def grayWorld(img):
    tmp = img.astype('float64')
    meanR = np.mean(tmp[:, :, 2])
    meanG = np.mean(tmp[:, :, 1])
    meanB = np.mean(tmp[:, :, 0])

    mean = (meanR + meanG + meanB) / 3
    tmp[:, :, 2] *= mean / (meanR + EPSILON)
    tmp[:, :, 1] *= mean / (meanG + EPSILON)
    tmp[:, :, 0] *= mean / (meanB + EPSILON)

    return np.clip(tmp, 0, 255).astype('uint8')
