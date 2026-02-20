import numpy as np
import matplotlib.pyplot as plt

def showHystogramSingleChannelImages(img, gaps=256, start=0, stop=256):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 2: raise ValueError('The argument \'img\' must have two dimensions exactly - \'height\' and \'weight\'')

    plt.hist(img.flatten(), bins=gaps, range=(start, stop), color='black', rwidth=1)
    plt.xlabel('Level of brightness')
    plt.ylabel('Number of pixels')
    plt.xlim(start, stop)
    plt.grid(True)
    plt.show()

def showHystogramThreeChannelsImages(img, gaps=256, start=0, stop=256):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')
    if img.ndim != 3: raise ValueError('The argument \'img\' must have three dimensions exactly - \'height\', \'weight\' and \'channels\'')

    RGB: tuple[str, str, str] = ('red', 'blue', 'green')

    for i, c in enumerate(RGB):
        plt.hist(img[:, :, i].flatten(), bins=gaps, range=(start, stop), alpha=0.5, color=c, rwidth=1)

    plt.xlabel('Level of brightness')
    plt.ylabel('Number of pixels')
    plt.xlim(start, stop)
    plt.grid(True)
    plt.show()

def FourierDecomposition(img):
    if img.size == 0: raise ValueError('The argument \'img\' must be not empty')

    tmp = img.astype('float64')
    four = np.fft.fftshift(np.fft.fft2(tmp))
    
    amplitude = np.abs(four)
    
    logs = np.log(amplitude + 1)
    
    plt.imshow(logs, cmap='Blues')
    plt.title('Fourier\'s Decomposition (log)')
    plt.show()