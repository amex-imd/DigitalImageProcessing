import numpy as np
import matplotlib.pyplot as plt

def showHystogramSingleChannelImages(img, gaps=256, start=0, stop=256, isNormalized = True):
    plt.hist(img.flatten(), density=isNormalized, bins=gaps, range=(start, stop), color='black', rwidth=1)
    plt.xlabel('Level of brightness')
    plt.ylabel('Number of pixels')
    plt.xlim(start, stop)
    plt.grid(True)
    plt.show()

def showHystogramThreeChannelsImages(img, gaps=256, start=0, stop=256, isNormalized = True):
    RGB: tuple[str, str, str] = ('red', 'blue', 'green')

    for i, c in enumerate(RGB): plt.hist(img[:, :, i].flatten(), density=isNormalized, bins=gaps, range=(start, stop), alpha=0.5, color=c, rwidth=1)

    plt.xlabel('Level of brightness')
    plt.ylabel('Number of pixels')
    plt.xlim(start, stop)
    plt.grid(True)
    plt.show()

def FourierDecomposition(img):
    tmp = img.astype('float64')
    four = np.fft.fftshift(np.fft.fft2(tmp))
    
    amplitude = np.abs(four)
    
    logs = np.log(1+amplitude)
    
    plt.imshow(logs, cmap='Blues')
    plt.title('Fourier\'s Decomposition (log)')
    plt.show()