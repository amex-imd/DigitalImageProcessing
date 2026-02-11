# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import DIP

def main() -> None:
    img = cv2.imread('imgs/income.jpg')
    res = DIP.linearTransformation(img=img, alpha=2,beta=1)
    cv2.imwrite('imgs/outcome.jpg', res)


    print('Hello, world!')

main()