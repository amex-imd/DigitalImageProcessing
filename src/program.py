# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
import DIP

def main() -> None:
    img = cv2.imread('imgs/income.jpg')
    res = DIP.addPoissonNoise(img, epp=100)
    cv2.imwrite('imgs/noise.jpg', res) 
    res = DIP.GaussianFilterThreeChannelsImages(res, kernel_size=3, sigma=10)
    cv2.imwrite('imgs/outcome.jpg', res) 
    print('Hello, world!')
    

main()