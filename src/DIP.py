# МОДУЛЬ ДЛЯ DIGITAL IMAGE PROCESSING
import numpy as np


def linearTransformation(img, alpha: float = 1.0, beta: float = 0.0):
    res = alpha * img.astype(np.float64) + beta
    res = np.clip(res, 0, 255).astype('uint8')
    return res
