# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
import DIP
import mathtools as mt
from scipy import signal

def main() -> None:



    img = cv2.imread('imgs/income.jpg')
    
    img = DIP.SobelFilterThreeChannelsImages(img, 5)
    cv2.imwrite('imgs/outcome.jpg', img)
    print('Hello, world!')

main()

