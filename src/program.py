# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
import DIP

def main() -> None:
    img = cv2.imread('imgs/income.jpg')
    img = DIP.addSaltAndPepperNoise(img, 0.1, 0.1)
    cv2.imwrite('imgs/temp.jpg', img) 
    res = DIP.medianFilterThreeChannelsImages(img, 5)
    cv2.imwrite('imgs/outcome.jpg', res) 
    print('Hello, world!')
    

main()