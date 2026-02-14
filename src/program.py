# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
import DIP

def main() -> None:
    img = cv2.imread('imgs/income.jpg')
    noise = DIP.addSaltAndPepperNoise(img=img, saltProb=0.01, pepperProb=0.03)
    res = DIP.medianFilterThreeChannelsImages(noise, kernel_size=3)
    cv2.imwrite('imgs/outcome.jpg', res)   
    cv2.imwrite('imgs/noise.jpg', noise)  
    print('Hello, world!')

    

main()