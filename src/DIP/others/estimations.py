import numpy as np
from DIP.others.constants import EPSILON

def MSE(first, second): # Mean Squared Error
    return np.mean(np.power(first.astype('float64') - second.astype('float64'), 2))

def PSNR(first, second): # Peak Signal-to-Noise Ratio
    dtypeInfo = np.iinfo(first.dtype)
    return 10 * np.log10(dtypeInfo.max * dtypeInfo.max / (MSE(first, second) + EPSILON)), dtypeInfo.max

def MAE(first, second): # Mean Absolute Error
    return np.mean(np.abs(first.astype('float64') - second.astype('float64')))

def SNR(src, proc): # Signal-to-Noise Ratio
    signalPower = np.mean(src.astype("float64") * src.astype("float64"))
    noise = src.astype("float64") - proc.astype("float64")
    noisePower = np.mean(noise * noise)

    if noisePower == 0: return float("inf")
    return 10 * np.log10(signalPower / noisePower)

def NRR(src, noise, proc): # Noise Reduction Rate
    tmp1 = np.mean((src.astype("float64") - noise.astype("float64")) ** 2)
    tmp2 = np.mean((src.astype(float) - proc.astype(float)) ** 2)
    if tmp1 == 0: return 1
    return max(0, 1-tmp2/tmp1)
