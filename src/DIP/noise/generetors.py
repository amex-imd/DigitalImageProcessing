import numpy as np

def addSaltAndPepperNoise(img, saltProb: float = 0.01, pepperProb: float = 0.01):
    if not(0 <= saltProb <= 1): raise ValueError('The argument \'saltProb\' must be bellow 0 and 1')
    if not(0 <= pepperProb <= 1): raise ValueError('The argument \'pepperProb\' must be bellow 0 and 1')
    if 1 - saltProb - pepperProb < 0: raise ValueError('The arguments \'saltProb\' and \'pepperProb\' are out of range')

    res = np.copy(img)

    randomMask = np.random.uniform(low=0, high=1, size=img.shape[:2])
    saltMask = randomMask < saltProb
    pepperMask = (randomMask >= saltProb) & (randomMask < saltProb + pepperProb)

    if img.ndim == 3:
        res[saltMask] = (255, 255, 255)
        res[pepperMask] = (0, 0, 0)
    elif img.ndim == 2:
        res[saltMask] = 255
        res[pepperMask] = 0

    return res

def addGaussianNoise(img, a: float = 0, sigma: float = 10):
    if sigma < 0: raise ValueError('The argument \'img\' must be equal to or greater than 0')

    noise = np.random.normal(loc=a, scale=sigma, size=img.shape)
    tmp = img.astype('float64') + noise

    return np.clip(tmp, 0, 255).astype('uint8')

def addUniformNoise(img, beg: int = -10, end: int = 10):
    if beg >= end: raise ValueError('The argument \'beg\' must be less than the argument \'end\'')

    noise = np.random.uniform(low=beg, high=end, size=img.shape)
    tmp = img.astype('float64') + noise

    return np.clip(tmp, 0, 255).astype('uint8')

def addRayleighNoise(img, sigma: float = 10):
    if sigma <= 0: raise ValueError('The argument \'img\' must be greater than 0')

    noise = np.random.rayleigh(scale=sigma, size=img.shape)
    tmp = img.astype('float64') + noise

    return np.clip(tmp, 0, 255).astype('uint8')

def addGammaNoise(img, k: float = 4, o: float = 4):
    if o <= 0: raise ValueError('The argument \'o\' must be greater than 0')
    if k <= 0: raise ValueError('The argument \'k\' must be greater than 0')

    noise = np.random.gamma(scale=k, shape=o, size=img.shape)
    tmp = img.astype('float64') + noise

    return np.clip(tmp, 0, 255).astype('uint8')

def addExponentialNoise(img, lam: float=10):
    if lam <= 0: raise ValueError('The argument \'lam\' must be greater than 0')

    noise = np.random.exponential(scale=lam, size=img.shape)
    tmp = img.astype('float64') + noise

    return np.clip(tmp, 0, 255).astype('uint8')

def addSinusoidalNoise(img, frequency: float = 10, amplitude: float = 0.1, angle: float = np.pi / 2, phase: float = 0):
    if frequency < 0: raise ValueError('The argument \'frequency\' must be equal to or greater than 0')

    tmp = img.astype('float64')
    x, y = np.arange(img.shape[1]), np.arange(img.shape[0])
    xx, yy = np.meshgrid(x, y)

    ax = xx * np.cos(angle) - yy * np.sin(angle)
    noise = amplitude * np.sin(2*np.pi*frequency*ax + phase)
    if len(img.shape) == 3: tmp += noise[:, :, np.newaxis]
    else: tmp += noise
    
    return np.clip(tmp, 0, 255).astype('uint8')