# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
import DIP

def main() -> None:
    img = cv2.imread('imgs/income.jpg', cv2.IMREAD_GRAYSCALE)
    DIP.FourierDecomposition(img)
    img = DIP.addSinusoidalNoise(img, 200, 100, 100, 0)
    DIP.FourierDecomposition(img)
    cv2.imwrite('imgs/temp.jpg', img) 
    print('Hello, world!')
    

main()