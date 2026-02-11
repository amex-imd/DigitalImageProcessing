# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
import DIP

def main() -> None:
    print('Hello, world!')
    img = cv2.imread('imgs/income.jpg', cv2.IMREAD_GRAYSCALE)
    mrx = np.array([[0, 1, 0, 0, 8],
                   [2, 128, 5, 1, 59],
                   [9, 0, 201, 3, 3],
                   [4, 1, 1, 1, 0],
                   [7, 0, 0, 0, 7]])
    res = DIP.meanFilterSingleChannelImages(img=img, kernel_size=5)
    cv2.imwrite('imgs/outcome.jpg', res)   

    



    print('Hello, world!')

    

main()