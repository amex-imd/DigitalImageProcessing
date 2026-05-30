import numpy as np
import DIP.others.extra_tools as ext
from DIP.others.constants import EPSILON, MAX_PIXEL_VALUE, MIN_MAX_PIXEL_VALUE

def bilateralFilterSingleChannelImages(img, d=9, sigmaColor=75, sigmaSpace=75, mode="edge"):
    if img.ndim != 2: raise ValueError("The argument \"img\" must have two dimensions")
    if d % 2 == 0: d += 1
    radius = d // 2
    h, w = img.shape
    img_padded = np.pad(img.astype("float64"), pad_width=radius, mode=mode)
    res = np.zeros_like(img, dtype="uint8")
    
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(x, y)

    spatial_weights = np.exp(-(xx*xx + yy*yy) / (2 * sigmaSpace * sigmaSpace))

    for i in range(h):
        for j in range(w):
            center_x, center_y = i + radius, j + radius
            center_val = img_padded[center_x, center_y]
            window = img_padded[center_x - radius:center_x + radius + 1, center_y - radius:center_y + radius + 1]
            photometric_weights = np.exp(-((window - center_val) ** 2) / (2 * sigmaColor * sigmaColor))
            weights = spatial_weights * photometric_weights
            total_weight = np.sum(weights)

            if total_weight > 0:
                res[i, j] = np.clip(np.sum(weights * window) / total_weight, 0, 255).astype("uint8")
            else:
                res[i, j] = np.clip(center_val, 0, 255).astype("uint8")
    return res

def bilateralFilterThreeChannelImages(img, d=9, sigmaColor=75, sigmaSpace=75, mode="edge"):
    if img.ndim != 3: raise ValueError("The argument \"img\" must have three dimensions")
    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype="uint8")
    for i in range(c): res[:, :, i] = bilateralFilterSingleChannelImages(img[:, :, i], d, sigmaColor, sigmaSpace, mode)
    return res

def arithmeticMeanFilterSingleChannelImages(img, filterSize=3, mode="edge"):
    if img.ndim != 2: raise ValueError("The argument \"img\" must have two dimensions exactly - \"height\" and \"weight\"")
    if filterSize < 0: raise ValueError("The argument \"filterSize\" must be equal to or greater than 0")
    if filterSize % 2 == 0: raise ValueError("The argument \"filterSize\" must be odd")

    gap = filterSize // 2
    h, w = img.shape

    tmp = np.pad(array=img.astype("float64"), pad_width=gap, mode=mode)
    integral = ext.integralMatrix(mrx=tmp)
    
    rows1, cols1 = np.ogrid[0:h, 0:w]
    rows2, cols2 = rows1+filterSize-1, cols1+filterSize-1
    tmp1 = np.where((rows1 > 0) & (cols1 > 0), integral[rows1-1, cols1-1], 0) 
    tmp2 = np.where(rows1 > 0, integral[rows1-1, cols2], 0)
    tmp3 = np.where(cols1 > 0, integral[rows2, cols1-1], 0)
    tmp4 = integral[rows2, cols2]
    
    S = tmp4 - tmp2 - tmp3 + tmp1
    res = S / (filterSize * filterSize)

    return res.astype("uint8")

def arithmeticMeanFilterThreeChannelsImages(img, filterSize=3, mode="edge"):
    if img.ndim != 3: raise ValueError("The argument \"img\" must have three dimensions exactly - \"height\", \"weight\" and \"channels\"")
    if filterSize < 0: raise ValueError("The argument \"filterSize\" must be equal to or greater than 0")
    if filterSize % 2 == 0: raise ValueError("The argument \"filterSize\" must be odd")

    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype="uint8") # Using np.empty is better using np.zeroes because it doesn"t use memory fully

    for i in range(c): res[:, :, i] = arithmeticMeanFilterSingleChannelImages(img=img[:, :, i], filterSize=filterSize, mode=mode)

    return res

def geometricMeanFilterSingleChannelImages(img, filterSize=3, mode="edge"):
    if img.ndim != 2: raise ValueError("The argument \"img\" must have two dimensions exactly - \"height\" and \"weight\"")
    if filterSize < 0: raise ValueError("The argument \"filterSize\" must be equal to or greater than 0")
    if filterSize % 2 == 0: raise ValueError("The argument \"filterSize\" must be odd")

    gap = filterSize // 2
    h, w = img.shape

    tmp = np.pad(array=np.log(img.astype("float64") + EPSILON), pad_width=gap, mode=mode)
    integral = ext.integralMatrix(mrx=tmp)
    
    rows1, cols1 = np.ogrid[0:h, 0:w] # The final image size will be not changed
    rows2, cols2 = rows1+filterSize-1, cols1+filterSize-1
    term1 = np.where((rows1 > 0) & (cols1 > 0), integral[rows1-1, cols1-1], 0) 
    term2 = np.where(rows1 > 0, integral[rows1-1, cols2], 0)
    term3 = np.where(cols1 > 0, integral[rows2, cols1-1], 0)
    term4 = integral[rows2, cols2]
    
    S = term4 - term2 - term3 + term1
    res = np.exp(S / (filterSize * filterSize))

    return np.clip(res, MIN_MAX_PIXEL_VALUE, MAX_PIXEL_VALUE).astype("uint8")

def geometricMeanFilterThreeChannelsImages(img, filterSize=3, mode="edge"):
    if img.ndim != 3: raise ValueError("The argument \"img\" must have three dimensions exactly - \"height\", \"weight\" and \"channels\"")
    if filterSize < 0: raise ValueError("The argument \"filterSize\" must be equal to or greater than 0")
    if filterSize % 2 == 0: raise ValueError("The argument \"filterSize\" must be odd")

    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype="uint8") # Using np.empty is better using np.zeroes because it doesn"t use memory fully

    for i in range(c): res[:, :, i] = geometricMeanFilterSingleChannelImages(img=img[:, :, i], filterSize=filterSize, mode=mode)

    return res

def medianFilterSingleChannelImages(img, filterSize=3, mode="edge"):
    if img.ndim != 2: raise ValueError("The argument \"img\" must have two dimensions exactly - \"height\" and \"weight\"")
    if filterSize < 0: raise ValueError("The argument \"filterSize\" must be equal to or greater than 0")
    if filterSize % 2 == 0: raise ValueError("The argument \"filterSize\" must be odd")

    gap = filterSize // 2
    tmp = np.pad(array=img.astype("float64"), pad_width=gap, mode=mode)

    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(filterSize, filterSize))
    res = np.median(space, axis=(2, 3))

    return np.clip(res, MIN_MAX_PIXEL_VALUE, MAX_PIXEL_VALUE).astype("uint8") # The final image size will be not changed

def medianFilterThreeChannelsImages(img, filterSize: int = 3, mode: str = "edge"):
    if img.ndim != 3: raise ValueError("The argument \"img\" must have three dimensions exactly - \"height\", \"weight\" and \"channels\"")
    if filterSize < 0: raise ValueError("The argument \"filterSize\" must be equal to or greater than 0")
    if filterSize % 2 == 0: raise ValueError("The argument \"filterSize\" must be odd")

    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype="uint8") # Using np.empty is better using np.zeroes because it doesn"t use memory fully

    for i in range(c): res[:, :, i] = medianFilterSingleChannelImages(img=img[:, :, i], filterSize=filterSize, mode=mode)

    return res

def midpointFilterSingleChannelImages(img, filterSize=3, mode="edge"):
    if img.ndim != 2: raise ValueError("The argument \"img\" must have two dimensions exactly - \"height\" and \"weight\"")
    if filterSize < 0: raise ValueError("The argument \"filterSize\" must be equal to or greater than 0")
    if filterSize % 2 == 0: raise ValueError("The argument \"filterSize\" must be odd")

    gap = filterSize // 2
    tmp = np.pad(array=img.astype("float64"), pad_width=gap, mode=mode)

    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(filterSize, filterSize))
    maxVals = np.max(space, axis=(2, 3))
    minVals = np.min(space, axis=(2, 3))
    res = (minVals + maxVals) / 2

    return res.astype("uint8")

def midpointFilterThreeChannelsImages(img, filterSize=3, mode="edge"):
    if img.ndim != 3: raise ValueError("The argument \"img\" must have three dimensions exactly - \"height\", \"weight\" and \"channels\"")
    if filterSize < 0: raise ValueError("The argument \"filterSize\" must be equal to or greater than 0")
    if filterSize % 2 == 0: raise ValueError("The argument \"filterSize\" must be odd")

    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype="uint8") # Using np.empty is better using np.zeroes because it doesn"t use memory fully

    for i in range(c): res[:, :, i] = midpointFilterSingleChannelImages(img=img[:, :, i], filterSize=filterSize, mode=mode)
    
    return res

def GaussianFilterSingleChannelImages(img, filterSize=3, sigma=1, mode="edge"):
    if img.ndim != 2: raise ValueError("The argument \"img\" must have two dimensions exactly - \"height\" and \"weight\"")
    if filterSize < 0: raise ValueError("The argument \"filterSize\" must be equal to or greater than 0")
    if filterSize % 2 == 0: raise ValueError("The argument \"filterSize\" must be odd")
    
    gap = filterSize // 2 # also center
    tmp = np.pad(array=img.astype("float64"), pad_width=gap, mode=mode)

    filter = ext.GaussianFilter(filterSize, sigma)
    space = np.lib.stride_tricks.sliding_window_view(tmp, (filterSize, filterSize))

    res = np.tensordot(space, filter, axes=((2, 3), (0, 1)))
    return np.clip(res, MIN_MAX_PIXEL_VALUE, MAX_PIXEL_VALUE).astype("uint8") # The final image size will be not changed

def GaussianFilterThreeChannelsImages(img, filterSize=3, sigma=1, mode="edge"):
    if img.ndim != 3: raise ValueError("The argument \"img\" must have three dimensions exactly - \"height\", \"weight\" and \"channels\"")
    if filterSize < 0: raise ValueError("The argument \"filterSize\" must be equal to or greater than 0")
    if filterSize % 2 == 0: raise ValueError("The argument \"filterSize\" must be odd")

    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype="uint8") # Using np.empty is better using np.zeroes because it doesn"t use memory fully

    for i in range(c): res[:, :, i] = GaussianFilterSingleChannelImages(img=img[:, :, i], filterSize=filterSize, sigma=sigma, mode=mode)

    return res

def meanFrames(imgLst):
    tmp = [x.astype("float64") for x in imgLst]

    S = np.zeros_like(imgLst[0], dtype="float64")
    for x in tmp: S += x

    res = S / len(imgLst)

    return np.clip(res, MIN_MAX_PIXEL_VALUE, MAX_PIXEL_VALUE).astype("uint8")