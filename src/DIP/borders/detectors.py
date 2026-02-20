import numpy as np
import DIP.others.extra_tools as ext


def LaplaceFilterSingleChannelImages(img, filterSize: int = 3, mode: str = 'edge'):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 2: raise ValueError('The argument \'img\' must have two dimensions exactly - \'height\' and \'weight\'')
    if filterSize < 0: raise ValueError('The argument \'filterSize\' must be equal to or greater than 0')
    if filterSize % 2 == 0: raise ValueError('The argument \'filterSize\' must be odd')

    res = np.empty(shape=img.shape, dtype=img.dtype)
    
    gap = filterSize // 2 # also center
    tmp = np.pad(array=img.astype('float64'), pad_width=gap, mode=mode)
    
    filter = ext.LaplaceFilter(filterSize=filterSize)
    space = np.lib.stride_tricks.sliding_window_view(tmp, (filterSize, filterSize))

    res = np.tensordot(space, filter, axes=((2, 3), (0, 1)))
    res = res / np.max(res) * 255
    return np.clip(res, 0, 255).astype('uint8')

def LaplaceFilterThreeChannelsImages(img, filterSize: int = 3, mode: str = 'edge'):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 3: raise ValueError('The argument \'img\' must have three dimensions exactly - \'height\', \'weight\' and \'channels\'')
    if filterSize < 0: raise ValueError('The argument \'filterSize\' must be equal to or greater than 0')
    if filterSize % 2 == 0: raise ValueError('The argument \'filterSize\' must be odd')

    h, w, c = img.shape
    gap = filterSize // 2
    
    tmp = np.pad(array=img.astype('float64'), 
                 pad_width=((gap, gap), (gap, gap), (0, 0)), 
                 mode=mode)
    
    filter = ext.LaplaceFilter(filterSize=filterSize)
    print(filter)

    filter = filter[:, :, np.newaxis]
    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(filterSize, filterSize, c))
    space = space.reshape(h, w, filterSize, filterSize, c)
    
    res = np.sum(space * filter, axis=(2, 3))
    res = res / np.max(res) * 255
    return np.clip(res, 0, 255).astype('uint8')

def SobelFilterSingleChannelImages(img, filterSize: int = 3, mode: str = 'edge', direction: str = 'xy'):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 2: raise ValueError('The argument \'img\' must have two dimensions exactly - \'height\' and \'weight\'')
    if filterSize < 0: raise ValueError('The argument \'filterSize\' must be equal to or greater than 0')
    if filterSize % 2 == 0: raise ValueError('The argument \'filterSize\' must be odd')

    gap = filterSize // 2
    
    tmp = np.pad(array=img.astype('float64'), pad_width=gap, mode=mode)
    
    fx, fy = ext.SobelFilter(filterSize)
    
    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(filterSize, filterSize))
    
    if direction == 'x': res = np.tensordot(space, fx, axes=((2, 3), (0, 1)))
    elif direction == 'y': res = np.tensordot(space, fy, axes=((2, 3), (0, 1)))
    elif direction == 'xy':
        gx = np.tensordot(space, fx, axes=((2, 3), (0, 1)))
        gy = np.tensordot(space, fy, axes=((2,3), (0, 1)))
        res = np.sqrt(gx*gx + gy*gy)
    
    return np.clip(res, 0, 255).astype('uint8')

def SobelFilterThreeChannelsImages(img, filterSize: int = 3, mode: str = 'edge', direction: str = 'xy'):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 3: raise ValueError('The argument \'img\' must have three dimensions exactly - \'height\', \'weight\' and \'channels\'')
    if filterSize < 0: raise ValueError('The argument \'filterSize\' must be equal to or greater than 0')
    if filterSize % 2 == 0: raise ValueError('The argument \'filterSize\' must be odd')

    gap = filterSize // 2
    h, w, c = img.shape
    
    tmp = np.pad(array=img.astype('float64'), pad_width=((gap, gap), (gap, gap), (0, 0)), mode=mode)
    
    fx, fy = ext.SobelFilter(filterSize)

    fx = fx[:, :, np.newaxis]
    fy = fy[:, :, np.newaxis]
    
    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(filterSize, filterSize, c))
    space = space.reshape(h, w, filterSize, filterSize, c)
    
    if direction == 'x': res = np.sum(space * fx, axis=(2, 3))
    elif direction == 'y': res = np.sum(space * fy, axis=(2, 3))
    elif direction == 'xy':
        gx = np.sum(space * fx, axis=(2, 3))
        gy = np.sum(space * fy, axis=(2, 3))
        res = np.sqrt(gx*gx + gy*gy)
    
    return np.clip(res, 0, 255).astype('uint8')

def RobertsonSingleChannelImages(img, mode: str = 'edge', direction: str = 'xy'):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 2: raise ValueError('The argument \'img\' must have two dimensions exactly - \'height\' and \'weight\'')
    
    tmp = np.pad(array=img.astype('float64'), pad_width=1, mode=mode)
    
    fx, fy = ext.RobertsonFilter()
    
    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(2, 2)) # MAGIC NUMBERS
    
    if direction == 'x': res = np.tensordot(space, fx, axes=((2, 3), (0, 1)))
    elif direction == 'y': res = np.tensordot(space, fy, axes=((2, 3), (0, 1)))
    elif direction == 'xy':
        gx = np.tensordot(space, fx, axes=((2, 3), (0, 1)))
        gy = np.tensordot(space, fy, axes=((2,3), (0, 1)))
        res = np.sqrt(gx*gx + gy*gy)
    
    return np.clip(res, 0, 255).astype('uint8')

def RobertsonFilterThreeChannelsImages(img, mode: str = 'edge', direction: str = 'xy'):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 3: raise ValueError('The argument \'img\' must have three dimensions exactly - \'height\', \'weight\' and \'channels\'')

    h, w, c = img.shape
    
    tmp = np.pad(array=img.astype('float64'), pad_width=((1, 1), (1, 1), (0, 0)), mode=mode)
    
    fx, fy = ext.RobertsonFilter()

    fx = fx[:, :, np.newaxis]
    fy = fy[:, :, np.newaxis]
    
    space = np.lib.stride_tricks.sliding_window_view(tmp, window_shape=(2, 2, c))
    space = space.reshape(h+1, w+1, 2, 2, c)
    
    if direction == 'x': res = np.sum(space * fx, axis=(2, 3))
    elif direction == 'y': res = np.sum(space * fy, axis=(2, 3))
    elif direction == 'xy':
        gx = np.sum(space * fx, axis=(2, 3))
        gy = np.sum(space * fy, axis=(2, 3))
        res = np.sqrt(gx*gx + gy*gy)
    
    return np.clip(res, 0, 255).astype('uint8')