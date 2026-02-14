# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
import DIP

def main() -> None:
    img = cv2.imread('imgs/income.jpg')
    res = DIP.addGaussianNoise(img, a=0, sigma=25)
    cv2.imwrite('imgs/noise.jpg', res) 
    res = DIP.GaussianFilterThreeChannelsImages(res, kernel_size=3, sigma=25)
    cv2.imwrite('imgs/outcome.jpg', res) 
    print('Hello, world!')
    

main()