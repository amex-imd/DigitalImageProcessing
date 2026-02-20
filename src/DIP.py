import numpy as np

# Future

def minFilterSingleChannelImages(img, filterSize: int = 3, mode: str = 'edge'):
    gap = filterSize // 2
    tmp = np.pad(array=img.astype('float64'), pad_width=gap, mode=mode)

    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(filterSize, filterSize))
    res = np.min(space, axis=(2, 3))

    return res.astype('uint8')

def minFilterThreeChannelsImages(img, filterSize: int = 3, mode: str = 'edge'):
    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype='uint8') # Using np.empty is better using np.zeroes because it doesn't use memory fully

    for i in range(c): res[:, :, i] = minFilterSingleChannelImages(img=img[:, :, i], filterSize=filterSize, mode=mode)

    return res

def maxFilterSingleChannelImages(img, filterSize: int = 3, mode: str = 'edge'):
    gap = filterSize // 2
    tmp = np.pad(array=img.astype('float64'), pad_width=gap, mode=mode)

    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(filterSize, filterSize))
    res = np.max(space, axis=(2, 3))

    return res.astype('uint8')

def maxFilterThreeChannelsImages(img, filterSize: int = 3, mode: str = 'edge'):
    h, w, c = img.shape
    res = np.empty(shape=(h, w, c), dtype='uint8') # Using np.empty is better using np.zeroes because it doesn't use memory fully

    for i in range(c): res[:, :, i] = minFilterSingleChannelImages(img=img[:, :, i], filterSize=filterSize, mode=mode)

    return res