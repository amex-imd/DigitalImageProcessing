import numpy as np
from DIP.others.constants import EPSILON

def MSE(first, second):
    return np.mean(np.power(first.astype('float64') - second.astype('float64'), 2))

def PSNR(first, second):
    dtypeInfo = np.iinfo(first.dtype)
    return 10 * np.log10(dtypeInfo.max * dtypeInfo.max / (MSE(first, second) + EPSILON)), dtypeInfo.max

def RMSE(first, second):
    return np.sqrt(MSE(first, second))

def MAE(first, second):
    return np.mean(np.abs(first.astype('float64') - second.astype('float64')))

def PSNR_BB(src, noisy, proc):
    mse1 = MSE(src, noisy)
    mse2 = MSE(src, proc)

    if mse2 < mse1:
        improv = mse1 - mse2
        return 10 * np.log10(255*255/mse2) + improv
    else:
        return -10 * np.log10(255 * 255 / mse1)

def SNR(src, proc):
    signalPower = np.mean(src.astype("float64") * src.astype("float64"))
    noise = src.astype("float64") - proc.astype("float64")
    noisePower = np.mean(noise * noise)

    if noisePower == 0: return float("inf")
    return 10 * np.log10(signalPower / noisePower)

def NRR(src, noise, proc):
    tmp1 = np.mean((src.astype("float64") - noise.astype("float64")) ** 2)
    tmp2 = np.mean((src.astype(float) - proc.astype(float)) ** 2)
    if tmp1 == 0: return 1
    return max(0, 1-tmp2/tmp1)
