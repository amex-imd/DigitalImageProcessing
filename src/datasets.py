from DIP.noise.reductions import *
from DIP.noise.generetors import *
from DIP.others.estimations import *
from DIP.others.translators import *
import keras
import time
import pandas as pd
import os
import sklearn

DATASETS_NAME = ["cifar10", "cifar100", "lfw"]

PROCESSING_METHODS = {
    "ArithmeticMean3": {
        "func": arithmeticMeanFilterThreeChannelsImages,
        "args": {"filterSize": 3, "mode": "edge"},
    },
    "ArithmeticMean5": {
        "func": arithmeticMeanFilterThreeChannelsImages,
        "args": {"filterSize": 5, "mode": "edge"},
    },
    "ArithmeticMean7": {
        "func": arithmeticMeanFilterThreeChannelsImages,
        "args": {"filterSize": 7, "mode": "edge"},
    },

    "GeometricMean3": {
        "func": geometricMeanFilterThreeChannelsImages,
        "args": {"filterSize": 3, "mode": "edge"},
    },
    "GeometricMean5": {
        "func": geometricMeanFilterThreeChannelsImages,
        "args": {"filterSize": 5, "mode": "edge"},
    },
    "GeometricMean7": {
        "func": geometricMeanFilterThreeChannelsImages,
        "args": {"filterSize": 7, "mode": "edge"},
    },
    
    "Median3": {
        "func": medianFilterThreeChannelsImages,
        "args": {"filterSize": 3, "mode": "edge"},
    },
    "Median5": {
        "func": medianFilterThreeChannelsImages,
        "args": {"filterSize": 5, "mode": "edge"},
    },
    "Median7": {
        "func": medianFilterThreeChannelsImages,
        "args": {"filterSize": 7, "mode": "edge"},
    },
    
    "Midpoint3": {
        "func": midpointFilterThreeChannelsImages,
        "args": {"filterSize": 3, "mode": "edge"},
    },
    "Midpoint5": {
        "func": midpointFilterThreeChannelsImages,
        "args": {"filterSize": 5, "mode": "edge"},
    },
    "Midpoint7": {
        "func": midpointFilterThreeChannelsImages,
        "args": {"filterSize": 7, "mode": "edge"},
    },
    
    "Gaussian3_sigma1": {
        "func": GaussianFilterThreeChannelsImages,
        "args": {"filterSize": 3, "sigma": 1, "mode": "edge"},
    },
    "Gaussian3_sigma2": {
        "func": GaussianFilterThreeChannelsImages,
        "args": {"filterSize": 3, "sigma": 2, "mode": "edge"},
    },
    "Gaussian5_sigma1.5": {
        "func": GaussianFilterThreeChannelsImages,
        "args": {"filterSize": 5, "sigma": 1.5, "mode": "edge"},
    },
    "Gaussian5_sigma2": {
        "func": GaussianFilterThreeChannelsImages,
        "args": {"filterSize": 5, "sigma": 2, "mode": "edge"},
    },
    "Gaussian7_sigma2": {
        "func": GaussianFilterThreeChannelsImages,
        "args": {"filterSize": 7, "sigma": 2, "mode": "edge"},
    },
    
    "Min3": {
        "func": minFilterThreeChannelsImages,
        "args": {"filterSize": 3, "mode": "edge"},
    },
    "Min5": {
        "func": minFilterThreeChannelsImages,
        "args": {"filterSize": 5, "mode": "edge"},
    },
    "Min7": {
        "func": minFilterThreeChannelsImages,
        "args": {"filterSize": 7, "mode": "edge"},
    },
    
    "Max3": {
        "func": maxFilterThreeChannelsImages,
        "args": {"filterSize": 3, "mode": "edge"},
    },
    "Max5": {
        "func": maxFilterThreeChannelsImages,
        "args": {"filterSize": 5, "mode": "edge"},
    },
    "Max7": {
        "func": maxFilterThreeChannelsImages,
        "args": {"filterSize": 7, "mode": "edge"},
    },
    
    # TODO
    #"MeanFrames": {
    #    "func": meanFrames,
    #    "args": {},
    #}
}

NOISE_METHODS = {
    "gaussian": {
        "func": addGaussianNoise,
        "args": {"sigma": 25},
    },
    "salt_pepper": {
        "func": addSaltAndPepperNoise,
        "args": {"saltProb": 0.01, "pepperProb": 0.01},
    },
    "uniform": {
        "func": addUniformNoise,
        "args": {"beg": -20, "end": 20},
    },
    "rayleigh": {
        "func": addRayleighNoise,
        "args": {"sigma": 15},
    },
    "gamma": {
        "func": addGammaNoise,
        "args": {"k": 4, "o": 4},
    },
    "exponential": {
        "func": addExponentialNoise,
        "args": {"lam": 10},
    }
}

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'

def loadDataset(datasetName):
    match(datasetName):
        case "cifar10":
            (_, _), (x, y) = keras.datasets.cifar10.load_data()
            cats = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]
        case "cifar100":
            (_, _), (x, y) = keras.datasets.cifar100.load_data(label_mode="fine")
            cats = None # TODO
        case "lfw":
            lfw = sklearn.datasets.fetch_lfw_people(min_faces_per_person=20, resize=0.5)
            x = (lfw.images * 255).astype(np.uint8)
            x = np.stack([x, x, x], axis=-1)
            y = lfw.target
            cats = lfw.target_names
        case _:
            raise ValueError("Wrong the dataset name");

    return x, y, cats
        

def testDataset(datasetName, noiseFunc, noiseArgs, processFunc, processArgs, estimationFunc, filepath):
    res = []

    x, y, cats = loadDataset(datasetName)
    for idx, img in enumerate(x):
        print(f"Success: {idx/len(x) * 100}")
        try:
            noisy = noiseFunc(img, **noiseArgs)
            estimationNoise = estimationFunc(img, noisy)
            start = time.time()

            newImg = processFunc(noisy, **processArgs)
            period = (time.time() - start) * 1e3

            estimationImg = estimationFunc(img, newImg)

            if cats is not None and y is not None and idx < len(y):
                cat = cats[int(y[idx])]
            else:
                cat = "unknown"
        
            res.append(dict(imageIndex=idx, category=cat,
                            estimationNoise=estimationNoise, estimationProcessedImage=estimationImg,
                            improvement=estimationImg-estimationNoise, timeDelta=period))
        except Exception as e:
            print(f"Error: {e}")
            break

    df = pd.DataFrame(res)
    df.to_csv(filepath, index=False)

    
def main():
    for dsn in DATASETS_NAME:
        for nf, nfdesc in NOISE_METHODS.items():
            for pf, pfdesc in PROCESSING_METHODS.items():
                testDataset(datasetName=dsn, noiseFunc=nfdesc["func"], noiseArgs=nfdesc["args"],
                            processFunc=pfdesc["func"], processArgs=pfdesc["args"], estimationFunc=MSE,
                            filepath=f"reports/test_{dsn}_{nf}_{pf}.csv")

main()



          


    

