# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
import DIP

def main() -> None:
    img = cv2.imread('imgs/income.jpg')
    img = DIP.grayWorld(img)
    cv2.imwrite('imgs/outcome.jpg', img)
    print('Hello, world!')

main()