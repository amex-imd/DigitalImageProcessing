# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
import DIP

def main() -> None:
    img = cv2.imread('imgs/income.jpg')
    print(img.shape)
    res = DIP.histogramEqualizationThreeChannelImages(img=img)
    cv2.imwrite('imgs/outcome.jpg', res)

    mrx = np.array([[0, 1, 0, 0, 8],
                    [2, 128, 5, 1, 59],
                    [9, 0, 201, 3, 3],
                    [4, 1, 1, 1, 0],
                    [7, 0, 0, 0, 7]])
    
    ind = np.arange(5)
    print(mrx)
    integral = np.cumsum(np.cumsum(mrx, axis=0), axis=1)
    print(integral)



    print('Hello, world!')

    

main()