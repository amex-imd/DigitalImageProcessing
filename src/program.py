# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
import DIP

def main() -> None:
    img = cv2.imread('imgs/income.jpg')
    res = DIP.addSaltAndPepperNoise(img=img, saltProb=0.1, pepperProb=0.03)
    cv2.imwrite('imgs/outcome.jpg', res)   
    print('Hello, world!')

    

main()