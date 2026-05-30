import numpy as np
import matplotlib.pyplot as plt
import cv2
from DIP.others.constants import EPSILON, MAX_PIXEL_VALUE
from DIP.others.extra_tools import GaussianFilter
import scipy

def MSE(first, second): # Mean Squared Error
    return np.mean(np.power(first.astype("float64") - second.astype("float64"), 2))

def PSNR(first, second): # Peak Signal-to-Noise Ratio
    dtypeInfo = np.iinfo(first.dtype)
    return 10 * np.log10(dtypeInfo.max * dtypeInfo.max / (MSE(first, second) + EPSILON)), dtypeInfo.max

def MAE(first, second): # Mean Absolute Error
    return np.mean(np.abs(first.astype("float64") - second.astype("float64")))

def SNR(src, proc): # Signal-to-Noise Ratio
    signalPower = np.mean(src.astype("float64") * src.astype("float64"))
    noise = src.astype("float64") - proc.astype("float64")
    noisePower = np.mean(noise * noise)

    if noisePower == 0: return float("inf")
    return 10 * np.log10(signalPower / noisePower)

def NRR(src, noise, proc): # Noise Reduction Rate
    tmp1 = src.astype("float64")
    tmp2 = noise.astype("float64")
    tmp3 = proc.astype("float64")

    a = np.mean((tmp1 - tmp2) * (tmp1 - tmp2))
    b = np.mean((tmp1 - tmp3) * (tmp1 - tmp3))

    if a == 0: return 1
    return max(0, 1-b/a)

def SSIM(first, second, windowSize=9, sigma=1, mode="same"):
    L=MAX_PIXEL_VALUE
    K1=0.01
    K2=0.03

    C1 = (K1 * L) * (K1 * L)
    C2 = (K2 * L) * (K2 * L)

    tmp1 = first.astype("float64")
    tmp2 = second.astype("float64")

    if len(tmp1.shape) == 3:
        res = 0
        for c in range(tmp1.shape[2]):
            res += SSIM(tmp1[:,:,c], tmp2[:,:,c], windowSize, sigma, mode=mode)
        return res / tmp1.shape[2]
    
    gauss = GaussianFilter(windowSize, sigma)

    mu1 = scipy.signal.convolve2d(tmp1, gauss, mode=mode)
    mu2 = scipy.signal.convolve2d(tmp2, gauss, mode=mode)

    mu12 = mu1 * mu2
    mu1 *= mu1
    mu2 *= mu2

    sigma1 = scipy.signal.convolve2d(tmp1 * tmp1, gauss, mode=mode) - mu1
    sigma2 = scipy.signal.convolve2d(tmp2 * tmp2, gauss, mode=mode) - mu2
    sigma12 = scipy.signal.convolve2d(tmp1 * tmp2, gauss, mode=mode) - mu12

    res = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1 + mu2 + C1) * (sigma1 + sigma2 + C2))
    return np.mean(res)

def showHystogramSingleChannelImages(img, title="Figure", isNormalized = True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fig.suptitle(title, fontsize=14)
    
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis('off')
    
    axes[1].hist(img.flatten(), bins=256, range=(0, 256), 
                 density=isNormalized, color='black', alpha=0.7, rwidth=1)
    axes[1].set_xlabel('Level of brightness')
    axes[1].set_ylabel('Number of pixels' if not isNormalized else 'Density of pixels')
    axes[1].set_xlim(0, 256)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Hystogram')
    
    plt.tight_layout()
    plt.show()

def showHystogramThreeChannelsImages(img, title="Figure", isNormalized = True):
    RGB = ("red", "green",
           "blue")
    channels = ('Red channel (R)', 'Green channel (G)',
                'Blue channel (B)')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fig.suptitle(title, fontsize=14)

    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis('off')

    for i, (color, name) in enumerate(zip(RGB, channels)):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        axes[1].plot(hist, color=color, label=name, alpha=0.7, linewidth=1.5)
    
    axes[1].set_xlabel('Level of brightness')
    axes[1].set_ylabel('Number of pixels' if not isNormalized else 'Density of pixels')
    axes[1].set_xlim(0, 256)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Histogram')
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()