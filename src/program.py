# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import DIP

def main() -> None:
    img = cv2.imread('imgs/income.jpg')
    print(img.shape)
    res = DIP.histogramEqualizationThreeChannelImages(img=img)
    cv2.imwrite('imgs/outcome.jpg', res)


    print('Hello, world!')

main()