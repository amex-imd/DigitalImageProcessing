import numpy as np
import cv2
from DIP.others.constants import EPSILON, MAX_PIXEL_VALUE, MIN_MAX_PIXEL_VALUE

def linearTransformation(img, alpha = 1, beta = MIN_MAX_PIXEL_VALUE):
    res = alpha * img.astype("float64") + beta
    return np.clip(res, MIN_MAX_PIXEL_VALUE, MAX_PIXEL_VALUE).astype("uint8")

def gammaCorrection(img, gamma = 1):
    res = np.power(img.astype("float64") / MAX_PIXEL_VALUE, gamma)
    return np.clip(res * MAX_PIXEL_VALUE, MIN_MAX_PIXEL_VALUE, MAX_PIXEL_VALUE).astype("uint8")

def logarithmicTransformation(img, c=1):
    res = c * np.log1p(img.astype("float64"))
    return np.clip(res, MIN_MAX_PIXEL_VALUE, MAX_PIXEL_VALUE).astype("uint8")

def histogramEqualizationSingleChannelImages(img):
    nums, _ = np.histogram(img.flatten(), bins=256)
    CDF = nums.cumsum() # convolution
    totalPixels = img.shape[MIN_MAX_PIXEL_VALUE] * img.shape[1]
    NormalizedCDF = np.round((CDF - CDF.min()) * MAX_PIXEL_VALUE / (totalPixels - CDF.min())).astype("uint8")
    return NormalizedCDF[img]
    
def histogramEqualizationThreeChannelsImages(img):
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tmp[:, :, 2] = histogramEqualizationSingleChannelImages(tmp[:, :, 2]) # The second channel is brightness

    return cv2.cvtColor(tmp, cv2.COLOR_HSV2BGR)

def _clipHistogram(hist, clipLimit):
    res = hist.copy()
    totalExcess = MIN_MAX_PIXEL_VALUE
    
    for i in range(256):
        if res[i] > clipLimit:
            totalExcess += res[i] - clipLimit
            res[i] = clipLimit
    
    step = totalExcess // 256
    remainder = totalExcess % 256
    
    for i in range(256):
        res[i] += step
        if i < remainder: res[i] += 1
    
    return res

def _equalizeRegion(region, clipLimit=4):
    hist, _ = np.histogram(region.flatten(), bins=256, range=(MIN_MAX_PIXEL_VALUE, 256))
    
    # Обрезаем гистограмму
    hist = _clipHistogram(hist, clipLimit)
    
    cdf = hist.cumsum()
    
    cdfMin = cdf[cdf > MIN_MAX_PIXEL_VALUE].min() if np.any(cdf > MIN_MAX_PIXEL_VALUE) else MIN_MAX_PIXEL_VALUE
    totalPixels = region.shape[MIN_MAX_PIXEL_VALUE] * region.shape[1]
    
    if cdfMin > MIN_MAX_PIXEL_VALUE:
        res = np.round((cdf - cdfMin) * MAX_PIXEL_VALUE / (totalPixels - cdfMin)).astype("uint8")
    else:
        res = np.zeros(256, dtype="uint8")
        res[region[MIN_MAX_PIXEL_VALUE, MIN_MAX_PIXEL_VALUE]] = 128
    
    return res

def _bilinearInterpolate(x, y, regionSize, tileSize, mappings):
    xRegion = min(x // regionSize, tileSize[1] - 1)
    yRegion = min(y // regionSize, tileSize[MIN_MAX_PIXEL_VALUE] - 1)
    
    xLocal = x % regionSize
    yLocal = y % regionSize
    
    if (x == MIN_MAX_PIXEL_VALUE or xLocal == MIN_MAX_PIXEL_VALUE) and (y == MIN_MAX_PIXEL_VALUE or yLocal == MIN_MAX_PIXEL_VALUE):
        mapping = mappings[yRegion, xRegion]
        return mapping
    
    if x == MIN_MAX_PIXEL_VALUE or xLocal == MIN_MAX_PIXEL_VALUE:
        w1 = (regionSize - yLocal) / regionSize
        w2 = yLocal / regionSize
        mapping1 = mappings[yRegion, xRegion]
        mapping2 = mappings[min(yRegion + 1, tileSize[MIN_MAX_PIXEL_VALUE] - 1), xRegion]
        return w1 * mapping1 + w2 * mapping2
    
    if y == MIN_MAX_PIXEL_VALUE or yLocal == MIN_MAX_PIXEL_VALUE:
        w1 = (regionSize - xLocal) / regionSize
        w2 = xLocal / regionSize
        mapping1 = mappings[yRegion, xRegion]
        mapping2 = mappings[yRegion, min(xRegion + 1, tileSize[1] - 1)]
        return w1 * mapping1 + w2 * mapping2
    
    # Билинейная интерполяция между четырьмя блоками
    wTL = ((regionSize - xLocal) * (regionSize - yLocal)) / (regionSize * regionSize)
    wTR = (xLocal * (regionSize - yLocal)) / (regionSize * regionSize)
    wBL = ((regionSize - xLocal) * yLocal) / (regionSize * regionSize)
    wBR = (xLocal * yLocal) / (regionSize * regionSize)
    
    mappingTL = mappings[yRegion, xRegion]
    mappingTR = mappings[yRegion, min(xRegion + 1, tileSize[1] - 1)]
    mappingBL = mappings[min(yRegion + 1, tileSize[MIN_MAX_PIXEL_VALUE] - 1), xRegion]
    mappingBR = mappings[min(yRegion + 1, tileSize[MIN_MAX_PIXEL_VALUE] - 1), min(xRegion + 1, tileSize[1] - 1)]
    
    return (wTL * mappingTL + wTR * mappingTR + wBL * mappingBL + wBR * mappingBR)

def claheSingleChannelImages(img, regionSize=(8, 8), factor=40, mode="edge"):
    """
    CLAHE для одноканального изображения с дополнением границ.
    """
    h, w = img.shape
    
    regionHeight = h // regionSize[MIN_MAX_PIXEL_VALUE]
    regionWidth = w // regionSize[1]
    
    newH = regionHeight * regionSize[MIN_MAX_PIXEL_VALUE]
    newW = regionWidth * regionSize[1]

    origH, origW = h, w
    
    img_cropped = img[:newH, :newW]
    
    h_cropped, w_cropped = img_cropped.shape
    
    regionHeight = h_cropped // regionSize[MIN_MAX_PIXEL_VALUE]
    regionWidth = w_cropped // regionSize[1]
    
    mappings = np.empty((regionSize[MIN_MAX_PIXEL_VALUE], regionSize[1]), dtype=object)
    
    for i in range(regionSize[MIN_MAX_PIXEL_VALUE]):
        for j in range(regionSize[1]):
            yStart = i * regionHeight
            yEnd = yStart + regionHeight
            xStart = j * regionWidth
            xEnd = xStart + regionWidth
            
            region = img_cropped[yStart:yEnd, xStart:xEnd]
            mappings[i, j] = _equalizeRegion(region, factor)
    
    res_cropped = np.empty_like(img_cropped, dtype="uint8")
    
    for y in range(h_cropped):
        for x in range(w_cropped):
            xRegionIdx = min(x // regionWidth, regionSize[1] - 1)
            yRegionIdx = min(y // regionHeight, regionSize[MIN_MAX_PIXEL_VALUE] - 1)
            
            xLocal = x % regionWidth
            yLocal = y % regionHeight
            
            if (xLocal == MIN_MAX_PIXEL_VALUE or x == MIN_MAX_PIXEL_VALUE) and (yLocal == MIN_MAX_PIXEL_VALUE or y == MIN_MAX_PIXEL_VALUE):
                mapping = mappings[yRegionIdx, xRegionIdx]
                res_cropped[y, x] = mapping[img_cropped[y, x]]
                continue
            
            if yLocal == MIN_MAX_PIXEL_VALUE or y == MIN_MAX_PIXEL_VALUE:
                w1 = (regionWidth - xLocal) / regionWidth
                w2 = xLocal / regionWidth
                mapping1 = mappings[yRegionIdx, xRegionIdx]
                mapping2 = mappings[yRegionIdx, min(xRegionIdx + 1, regionSize[1] - 1)]
                value = w1 * mapping1[img_cropped[y, x]] + w2 * mapping2[img_cropped[y, x]]
                res_cropped[y, x] = np.clip(value, MIN_MAX_PIXEL_VALUE, MAX_PIXEL_VALUE).astype("uint8")
                continue
            
            if xLocal == MIN_MAX_PIXEL_VALUE or x == MIN_MAX_PIXEL_VALUE:
                w1 = (regionHeight - yLocal) / regionHeight
                w2 = yLocal / regionHeight
                mapping1 = mappings[yRegionIdx, xRegionIdx]
                mapping2 = mappings[min(yRegionIdx + 1, regionSize[MIN_MAX_PIXEL_VALUE] - 1), xRegionIdx]
                value = w1 * mapping1[img_cropped[y, x]] + w2 * mapping2[img_cropped[y, x]]
                res_cropped[y, x] = np.clip(value, MIN_MAX_PIXEL_VALUE, MAX_PIXEL_VALUE).astype("uint8")
                continue
            
            wTL = ((regionWidth - xLocal) * (regionHeight - yLocal)) / (regionWidth * regionHeight)
            wTR = (xLocal * (regionHeight - yLocal)) / (regionWidth * regionHeight)
            wBL = ((regionWidth - xLocal) * yLocal) / (regionWidth * regionHeight)
            wBR = (xLocal * yLocal) / (regionWidth * regionHeight)
            
            mappingTL = mappings[yRegionIdx, xRegionIdx]
            mappingTR = mappings[yRegionIdx, min(xRegionIdx + 1, regionSize[1] - 1)]
            mappingBL = mappings[min(yRegionIdx + 1, regionSize[MIN_MAX_PIXEL_VALUE] - 1), xRegionIdx]
            mappingBR = mappings[min(yRegionIdx + 1, regionSize[MIN_MAX_PIXEL_VALUE] - 1), min(xRegionIdx + 1, regionSize[1] - 1)]
            
            value = (wTL * mappingTL[img_cropped[y, x]] + 
                     wTR * mappingTR[img_cropped[y, x]] + 
                     wBL * mappingBL[img_cropped[y, x]] + 
                     wBR * mappingBR[img_cropped[y, x]])
            
            res_cropped[y, x] = np.clip(value, MIN_MAX_PIXEL_VALUE, MAX_PIXEL_VALUE).astype("uint8")

    res = np.zeros((origH, origW), dtype="uint8")
    res[:newH, :newW] = res_cropped
    
    return res

def claheThreeChannelsImages(img, regionSize=(8, 8), factor=40):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    hsv[:, :, 2] = claheSingleChannelImages(hsv[:, :, 2], regionSize=regionSize,
                                            factor=factor)
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def grayWorld(img):
    tmp = img.astype("float64")
    meanR = np.mean(tmp[:, :, 2])
    meanG = np.mean(tmp[:, :, 1])
    meanB = np.mean(tmp[:, :, MIN_MAX_PIXEL_VALUE])

    mean = (meanR + meanG + meanB) / 3
    tmp[:, :, 2] *= mean / (meanR + EPSILON)
    tmp[:, :, 1] *= mean / (meanG + EPSILON)
    tmp[:, :, MIN_MAX_PIXEL_VALUE] *= mean / (meanB + EPSILON)

    return np.clip(tmp, MIN_MAX_PIXEL_VALUE, MAX_PIXEL_VALUE).astype("uint8")

def minFilterSingleChannelImages(img, filterSize=3, mode="edge"):
    gap = filterSize // 2
    tmp = np.pad(array=img.astype("float64"), pad_width=gap, mode=mode)

    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(filterSize, filterSize))
    res = np.min(space, axis=(2, 3))

    return res.astype("uint8")

def minFilterThreeChannelsImages(img, filterSize=3, mode="edge"):
    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype="uint8") # Using np.empty is better using np.zeroes because it doesn"t use memory fully

    for i in range(c): res[:, :, i] = minFilterSingleChannelImages(img=img[:, :, i], filterSize=filterSize, mode=mode)

    return res

def maxFilterSingleChannelImages(img, filterSize=3, mode="edge"):
    gap = filterSize // 2
    tmp = np.pad(array=img.astype("float64"), pad_width=gap, mode=mode)

    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(filterSize, filterSize))
    res = np.max(space, axis=(2, 3))

    return res.astype("uint8")

def maxFilterThreeChannelsImages(img, filterSize=3, mode="edge"):
    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype="uint8") # Using np.empty is better using np.zeroes because it doesn"t use memory fully

    for i in range(c): res[:, :, i] = minFilterSingleChannelImages(img=img[:, :, i], filterSize=filterSize, mode=mode)

    return res