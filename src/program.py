# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
import DIP

def main() -> None:
    img = cv2.imread('imgs/income.jpg', cv2.IMREAD_GRAYSCALE)
    
    res = DIP.addRayleighNoise(img, 100)
    cv2.imwrite('imgs/tmp.jpg', res)
    res = DIP.geometricMeanFilterSingleChannelImages(res, kernelSize=5)
    cv2.imwrite('imgs/outcome.jpg', res) 
    print('Hello, world!')
    

main()