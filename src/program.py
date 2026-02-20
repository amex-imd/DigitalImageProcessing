# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
from DIP.noise.generetors import addGammaNoise, addExponentialNoise, addGaussianNoise, addRayleighNoise, addSaltAndPepperNoise, addSinusoidalNoise, addUniformNoise
from DIP.noise.reductions import arithmeticMeanFilterThreeChannelsImages, geometricMeanFilterThreeChannelsImages, medianFilterThreeChannelsImages, GaussianFilterThreeChannelsImages, midpointFilterThreeChannelsImages

def main() -> None:


    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        cv2.imshow('1st', frame)

        img = midpointFilterThreeChannelsImages(frame, filterSize=3)

       # img = cv2.boxFilter(frame, ddepth=cv2.CV_64F, ksize=(3,3))
        #img = np.clip(img, 0, 255).astype(np.uint8)
        
        cv2.imshow('2nd', np.clip(img, 0, 255).astype('uint8'))
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()
main()

