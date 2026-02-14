# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
import DIP

def main() -> None:
    img = cv2.imread('imgs/income.jpg')
    res = DIP.GaussianFilterThreeChannelsImages(img, kernel_size=9)
    cv2.imwrite('imgs/outcome.jpg', res)   
    print('Hello, world!')

    

main()