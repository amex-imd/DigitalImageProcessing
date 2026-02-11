# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import DIP

def main() -> None:
    img = cv2.imread('imgs/income.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('imgs/tmp.jpg', img)
    res = DIP.histogramEqualization(img=img)
    cv2.imwrite('imgs/outcome.jpg', res)


    print('Hello, world!')

main()