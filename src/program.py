# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
import DIP
import mathtools as mt
from scipy import signal
def main() -> None:



    img = cv2.imread('imgs/income.jpg', cv2.IMREAD_GRAYSCALE)
    
    img = DIP.LaplaceFilterSingleChannelImages(img, kernelSize=3)

    # img = cv2.Laplacian(img, ddepth=cv2.CV_64F)
    cv2.imwrite('imgs/outcome.jpg', img)
    print('Hello, world!')

main()

