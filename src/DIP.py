# МОДУЛЬ ДЛЯ DIGITAL IMAGE PROCESSING
import numpy as np
import cv2
import mathtools as mt
import matplotlib.pyplot as plt
from typing import Tuple

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

def FourierDecomposition(img):
    tmp = img.astype('float64')
    four = np.fft.fftshift(np.fft.fft2(tmp))
    
    amplitude = np.abs(four)
    
    logs = np.log(amplitude + 1)
    
    plt.imshow(logs, cmap='gray')
    plt.title('Fourier\'s Decomposition (log)')
    plt.show()

def showHystogramSingleChannelImages(img, gaps=256, start=0, stop=256):
    plt.hist(img.flatten(), bins=gaps, range=(start, stop), color='black', rwidth=1)
    plt.xlabel('Level of brightness')
    plt.ylabel('Number of pixels')
    plt.xlim(start, stop)
    plt.grid(True)
    plt.show()

def showHystogramThreeChannelsImages(img, gaps=256, start=0, stop=256):
    RGB: Tuple[str, str, str] = ('red', 'blue', 'green')
    for i, c in enumerate(RGB):
        plt.hist(img[:, :, i].flatten(), bins=gaps, range=(start, stop), alpha=0.5, color=c, rwidth=1)
    plt.xlabel('Level of brightness')
    plt.ylabel('Number of pixels')
    plt.xlim(start, stop)
    plt.grid(True)
    plt.show()

def arithmeticMeanFilterSingleChannelImages(img, kernelSize: int = 3, mode: str = 'edge'):
    gap = kernelSize // 2
    h, w = img.shape

    tmp = np.pad(array=img.astype('float64'), pad_width=2*gap, mode=mode) # 2 * gap to avoid of going beyond borders of image
    integral = mt.integralMatrix(mrx=tmp)
    
    rows1, cols1 = np.ogrid[gap:h+gap, gap:w+gap] # The final image size will be not changed
    rows2, cols2 = rows1+kernelSize, cols1+kernelSize
    S = integral[rows1, cols1] + integral[rows2, cols2] - integral[rows1, cols2] - integral[rows2, cols1]
    res = S / (kernelSize * kernelSize)
    return res.astype('uint8')

def arithmeticMeanFilterThreeChannelsImages(img, kernelSize: int = 3, mode: str = 'edge'):
    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype='uint8') # Using np.empty is better using np.zeroes because it doesn't use full memory
    for i in range(c):
        res[:, :, i] = arithmeticMeanFilterSingleChannelImages(img=img[:, :, i], kernel_size=kernelSize, mode=mode)
    return res

def geometricMeanFilterSingleChannelImages(img, kernelSize: int = 3, mode: str = 'edge'):
    gap = kernelSize // 2
    h, w = img.shape

    tmp = np.pad(array=np.log(img.astype('float64') + 1e-12), pad_width=2*gap, mode=mode) # 2 * gap to avoid of going beyond borders of image
    integral = mt.integralMatrix(mrx=tmp)
    
    rows1, cols1 = np.ogrid[gap:h+gap, gap:w+gap] # The final image size will be not changed
    rows2, cols2 = rows1+kernelSize, cols1+kernelSize
    S = integral[rows1, cols1] + integral[rows2, cols2] - integral[rows1, cols2] - integral[rows2, cols1]
    res = np.exp(S / (kernelSize * kernelSize))
    return np.clip(res, 0, 255).astype('uint8')

def geometricMeanFilterThreeChannelsImages(img, kernelSize: int = 3, mode: str = 'edge'):
    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype='uint8') # Using np.empty is better using np.zeroes because it doesn't use full memory
    for i in range(c):
        res[:, :, i] = geometricMeanFilterSingleChannelImages(img=img[:, :, i], kernel_size=kernelSize, mode=mode)
    return res


def medianFilterSingleChannelImages(img, kernelSize: int = 3, mode: str = 'edge'):
    gap = kernelSize // 2
    tmp = np.pad(array=img.astype('float64'), pad_width=gap, mode=mode)
    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(kernelSize, kernelSize)) # for optimization

    res = np.median(space, axis=(2, 3))
    return np.clip(res, 0, 255).astype('uint8') # The final image size will be not changed

def medianFilterThreeChannelsImages(img, kernelSize: int = 3, mode: str = 'edge'):
    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype='uint8') # Using np.empty is better using np.zeroes because it doesn't use full memory
    for i in range(c):
        res[:, :, i] = medianFilterSingleChannelImages(img=img[:, :, i], kernelSize=kernelSize, mode=mode)
    return res

def minFilterSingleChannelImages(img, kernelSize: int = 3, mode: str = 'edge'):
    gap = kernelSize // 2
    tmp = np.pad(array=img.astype('float64'), pad_width=gap, mode=mode)
    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(kernelSize, kernelSize)) # for optimization
    res = np.min(space, axis=(2, 3))
    return res.astype('uint8')

def minFilterThreeChannelsImages(img, kernelSize: int = 3, mode: str = 'edge'):
    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype='uint8') # Using np.empty is better using np.zeroes because it doesn't use full memory
    for i in range(c):
        res[:, :, i] = minFilterSingleChannelImages(img=img[:, :, i], kernelSize=kernelSize, mode=mode)
    return res

def maxFilterSingleChannelImages(img, kernelSize: int = 3, mode: str = 'edge'):
    gap = kernelSize // 2
    tmp = np.pad(array=img.astype('float64'), pad_width=gap, mode=mode)
    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(kernelSize, kernelSize)) # for optimization
    res = np.max(space, axis=(2, 3))
    return res.astype('uint8')

def maxFilterThreeChannelsImages(img, kernelSize: int = 3, mode: str = 'edge'):
    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype='uint8') # Using np.empty is better using np.zeroes because it doesn't use full memory
    for i in range(c):
        res[:, :, i] = minFilterSingleChannelImages(img=img[:, :, i], kernelSize=kernelSize, mode=mode)
    return res

def midpointFilterSingleChannelImages(img, kernelSize: int = 3, mode: str = 'edge'):
    gap = kernelSize // 2
    tmp = np.pad(array=img.astype('float64'), pad_width=gap, mode=mode)
    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(kernelSize, kernelSize)) # for optimization
    maxVals = np.max(space, axis=(2, 3))
    minVals = np.min(space, axis=(2, 3))
    res = (minVals + maxVals) / 2
    return res.astype('uint8')

def midpointFilterThreeChannelsImages(img, kernelSize: int = 3, mode: str = 'edge'):
    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype='uint8') # Using np.empty is better using np.zeroes because it doesn't use full memory
    for i in range(c):
        res[:, :, i] = midpointFilterSingleChannelImages(img=img[:, :, i], kernelSize=kernelSize, mode=mode)
    return res


def GaussianFilterSingleChannelImages(img, kernelSize: int = 3, sigma: float = 1, mode: str = 'edge'):
    res = np.empty(shape=img.shape, dtype=img.dtype)
    
    gap = kernelSize // 2 # also center
    h, w = img.shape
    tmp = np.pad(array=img.astype('float64'), pad_width=gap, mode=mode)
    x = np.arange(start=-gap, stop=gap + 1)

    x, y = np.meshgrid(x, x)
    filter = np.exp(np.negative((np.multiply(x, x) + np.multiply(y, y))) / (2 * sigma * sigma))
    filter /= np.sum(filter)
    space = np.lib.stride_tricks.sliding_window_view(tmp, (kernelSize, kernelSize)) # for optimization

    res = np.tensordot(space, filter, axes=((2, 3), (0, 1)))
    return np.clip(res[gap:gap+h, gap:gap+w], 0, 255).astype('uint8') # The final image size will be not changed

def GaussianFilterThreeChannelsImages(img, kernelSize: int = 3, sigma: float = 1, mode: str = 'edge'):
    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype='uint8') # Using np.empty is better using np.zeroes because it doesn't use full memory
    for i in range(c):
        res[:, :, i] = GaussianFilterSingleChannelImages(img=img[:, :, i], kernel_size=kernelSize, sigma=sigma, mode=mode)
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

def addGaussianNoise(img, a: float = 0, sigma: float = 1):
    noise = np.random.normal(loc=a, scale=sigma, size=img.shape)
    return np.clip(img + noise, 0, 255).astype('uint8')

def addPoissonNoise(img, epp: float): # electorns per pixel
    tmp = img.astype('float64') / 255 * epp # brightness is proportional to the number of captured photons
    noise = np.random.poisson(lam=tmp)
    
    return np.clip(noise / epp * 255, 0, 255).astype('uint8')

def addUniformNoise(img, start: int = -10, stop: int = 10):
    noise = np.random.uniform(low=start, high=stop, size=img.shape)
    tmp = img.astype('float64') + noise
    return np.clip(tmp, 0, 255).astype('uint8')

def addRayleighNoise(img, sigma: float = 1):
    noise = np.random.rayleigh(scale=sigma, size=img.shape)
    tmp = img.astype('float64') + noise
    return np.clip(tmp, 0, 255).astype('uint8')

def addGammaNoise(img, k: float = 1, o: float = 1):
    noise = np.random.gamma(scale=k, shape=o, size=img.shape)
    tmp = img.astype('float64') + noise
    return np.clip(tmp, 0, 255).astype('uint8')

def LaplaceFilterSingleChannelImages(img, kernelSize: int = 3, mode: str = 'edge', isDiagonals: bool = False):
    res = np.empty(shape=img.shape, dtype=img.dtype)
    
    gap = kernelSize // 2 # also center
    tmp = np.pad(array=img.astype('float64'), pad_width=gap, mode=mode)
    x = np.arange(start=-gap, stop=gap + 1)

    x, y = np.meshgrid(x, x)
    filter = mt.LaplaceKernelWithDiagonals(kernelSize) if isDiagonals else mt.LaplaceKernelWithoutDiagonals(kernelSize)
    space = np.lib.stride_tricks.sliding_window_view(tmp, (kernelSize, kernelSize)) # for optimization

    res = np.tensordot(space, filter, axes=((2, 3), (0, 1)))
    return np.clip(res, 0, 255).astype('uint8')

def LaplaceFilterThreeChannelsImages(img, kernelSize: int = 3, mode: str = 'edge', isDiagonals: bool = False):
    h, w, c = img.shape
    gap = kernelSize // 2
    
    tmp = np.pad(array=img.astype('float64'), 
                 pad_width=((gap, gap), (gap, gap), (0, 0)), 
                 mode=mode)
    
    kernel = mt.LaplaceKernelWithDiagonals(kernelSize) if isDiagonals else mt.LaplaceKernelWithoutDiagonals(kernelSize)

    kernel = kernel[:, :, np.newaxis]
    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(kernelSize, kernelSize, c))
    space = space.reshape(h, w, kernelSize, kernelSize, c)
    
    res = np.sum(space * kernel, axis=(2, 3))
    return np.clip(res, 0, 255).astype('uint8')

def SobelFilterSingleChannelImages(img, kernelSize: int = 3, mode: str = 'edge', direction: str = 'xy'):
    gap = kernelSize // 2
    
    tmp = np.pad(array=img.astype('float64'), pad_width=gap, mode=mode)
    
    fx = mt.SobelFilterX(kernelSize)
    fy = mt.SobelFilterY(kernelSize)
    
    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(kernelSize, kernelSize))
    
    if direction == 'x': res = np.tensordot(space, fx, axes=((2, 3), (0, 1)))
    elif direction == 'y': res = np.tensordot(space, fy, axes=((2, 3), (0, 1)))
    elif direction == 'xy':
        gx = np.tensordot(space, fx, axes=((2, 3), (0, 1)))
        gy = np.tensordot(space, fy, axes=((2,3), (0, 1)))
        res = np.sqrt(gx*gx + gy*gy)
    
    return np.clip(res, 0, 255).astype('uint8')

def SobelFilterThreeChannelsImages(img, kernelSize: int = 3, mode: str = 'edge', direction: str = 'xy'):
    gap = kernelSize // 2
    h, w, c = img.shape
    
    tmp = np.pad(array=img.astype('float64'), pad_width=((gap, gap), (gap, gap), (0, 0)), mode=mode)
    
    fx = mt.SobelFilterX(kernelSize)
    fy = mt.SobelFilterY(kernelSize)

    fx = fx[:, :, np.newaxis]
    fy = fy[:, :, np.newaxis]
    
    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(kernelSize, kernelSize, c))
    space = space.reshape(h, w, kernelSize, kernelSize, c)
    
    if direction == 'x': res = np.sum(space * fx, axis=(2, 3))
    elif direction == 'y': res = np.sum(space * fy, axis=(2, 3))
    elif direction == 'xy':
        gx = np.sum(space * fx, axis=(2, 3))
        gy = np.sum(space * fy, axis=(2, 3))
        res = np.sqrt(gx*gx + gy*gy)
    
    return np.clip(res, 0, 255).astype('uint8')