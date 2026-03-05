import numpy as np
from DIP.others.constants import EPSILON

def MSE(first, second):
    return np.mean(np.power(first.astype('float64') - second.astype('float64'), 2))

def PSNR(first, second):
    dtypeInfo = np.iinfo(first.dtype)
    return 10 * np.log10(dtypeInfo.max * dtypeInfo.max / (MSE(first, second) + EPSILON)), dtypeInfo.max
