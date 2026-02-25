import numpy as np

def MSE(first, second) -> float:
    return np.mean(np.power(first.astype('float64') - second.astype('float64'), 2))
