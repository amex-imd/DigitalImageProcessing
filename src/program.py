# СКРИПТ ДЛЯ ДЕМОНСТРАЦИИ
import cv2
from DIP.noise.generetors import addGammaNoise, addExponentialNoise, addGaussianNoise, addRayleighNoise, addSaltAndPepperNoise, addSinusoidalNoise, addUniformNoise
from DIP.noise.reductions import arithmeticMeanFilterThreeChannelsImages, geometricMeanFilterThreeChannelsImages, medianFilterThreeChannelsImages, GaussianFilterThreeChannelsImages, midpointFilterThreeChannelsImages
from DIP.others.estimations import PSNR

def main() -> None:
    filepath = 'imgs/income.jpg'

    img = cv2.imread(filepath)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))
    res = clahe.apply(img)
    print(PSNR(img, res))
    print(PSNR(img, img))

main()

