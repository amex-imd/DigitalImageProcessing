# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
from DIP.noise.generetors import addGammaNoise, addExponentialNoise, addGaussianNoise, addRayleighNoise, addSaltAndPepperNoise, addSinusoidalNoise, addUniformNoise
from DIP.noise.reductions import arithmeticMeanFilterThreeChannelsImages, geometricMeanFilterThreeChannelsImages, medianFilterThreeChannelsImages, GaussianFilterThreeChannelsImages, midpointFilterThreeChannelsImages
from DIP.others.estimations import MSE
def main() -> None:
    filepath: str = 'imgs/income.jpg'

    img = cv2.imread(filepath)
    noise = addSaltAndPepperNoise(img)
    print(MSE(img, noise))
    res = medianFilterThreeChannelsImages(noise)
    print(MSE(img, res))

    print(MSE(img, img))

main()

