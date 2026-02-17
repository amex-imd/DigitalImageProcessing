# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
import DIP

def main() -> None:
    img = cv2.imread('imgs/income.jpg', cv2.IMREAD_GRAYSCALE)
    
    cv2.imwrite('imgs/tmp.jpg', img) 
    res = DIP.arithmeticMeanFilterSingleChannelImages(img, kernelSize=3)
    cv2.imwrite('imgs/outcome.jpg', res) 
    print('Hello, world!')
    

main()