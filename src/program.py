# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
import numpy as np
from DIP.noise.generetors import addGammaNoise, addExponentialNoise, addGaussianNoise, addRayleighNoise, addSaltAndPepperNoise, addSinusoidalNoise, addUniformNoise
from DIP.noise.reductions import arithmeticMeanFilterThreeChannelsImages, geometricMeanFilterThreeChannelsImages, medianFilterThreeChannelsImages, GaussianFilterThreeChannelsImages, midpointFilterThreeChannelsImages
from DIP.others.estimations import PSNR
def main() -> None:
    filepath: str = 'imgs/income.jpg'

    img = cv2.imread(filepath)
    noise = addSaltAndPepperNoise(img)
    print(PSNR(img, noise))
    res = medianFilterThreeChannelsImages(noise)
    print(PSNR(img, res))
    print(PSNR(img, img))

main()

