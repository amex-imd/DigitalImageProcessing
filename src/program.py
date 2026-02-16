# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
import DIP

def main() -> None:
    img = cv2.imread('imgs/income.jpg')
    
    DIP.showHystogramThreeChannelsImages(img)
    cv2.imwrite('imgs/tmp.jpg', img) 
    res = DIP.LaplaceFilterThreeChannelsImages(img, kernelSize=5)
    cv2.imwrite('imgs/outcome.jpg', res) 
    print('Hello, world!')
    

main()