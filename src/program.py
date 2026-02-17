# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
import DIP

def main() -> None:
    img = cv2.imread('imgs/income.jpg')
    n: int = 5
    imgLst = []
    while n > 0:
        tmp = DIP.addSaltAndPepperNoise(img, 0.05, 0.05)
        cv2.imwrite(f"imgs/{n}.jpg", tmp)
        imgLst.append(tmp)
        n -= 1
    
    img = DIP.meanFrames(imgLst)
    cv2.imwrite('imgs/outcome.jpg', img)
    print('Hello, world!')

main()