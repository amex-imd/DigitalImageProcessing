import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from DIP.noise.reductions import *
from DIP.noise.generetors import *
from DIP.others.estimations import *
from DIP.others.translators import *

import keras
import time
import pandas as pd
import sklearn
import seaborn.objects as so

DATASETS_NAME = ["cifar10", "cifar100", "lfw"]
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

def testAlgorithm(datasetName, filepath, processFunc, processFuncArgs,
                  noiseFunc, noiseFuncArgs, estimationFunc):
    try:
        x, y, cats = loadDataset(datasetName)
        data = []

        for idx, img in enumerate(x):
            noisy = noiseFunc(img, **noiseFuncArgs)
            estNoise = estimationFunc(img, noisy)
            start = time.time()

            newImg = processFunc(noisy, **processFuncArgs)
            period = (time.time() - start) * 1e3 # milliseconds

            estImg = estimationFunc(img, newImg)
            
            if cats is not None and y is not None and idx < len(y):
                cat = cats[int(y[idx])]
            else: cat = "unknown"
            data.append(dict(imageIndex=idx, category=cat, estimationNoise=estNoise, estimationProcessedImage=estImg, improvement=estNoise-estImg, timeDelta=period))
    except Exception as e:
        print(f"Error: {e}")

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

def drawGraphs(filepath):
    df = pd.read_csv(filepath)
    df["result"] = df["improvement"].apply(lambda x: "Good" if x > 0 else "Bad")
    result_counts = df["result"].value_counts().reset_index()
    result_counts.columns = ["result", "count"]

    plot = (so.Plot(result_counts, x="result", y="count")
            .add(so.Bar())
            .label(x="Результат", y="Количество"))
    plot.show()

    plot = (so.Plot(df, x="category", color="result")
            .add(so.Bar(), so.Count(), so.Dodge())
            .label(x="Результат", y="Количество"))
    plot.show()
    
def main():
    datasetName = "cifar10"
    processFunc = GaussianFilterThreeChannelsImages
    processFuncArgs = {"filterSize": 3, "mode": "edge"}

    noiseFunc = addGaussianNoise
    noiseFuncArgs = {"a": 0, "sigma": 5}

    estimationFunc = MSE

    filepath = f"reports/test_{datasetName}_{processFunc.__name__}_{noiseFunc.__name__}.csv"

    testAlgorithm(datasetName, filepath, processFunc, processFuncArgs,
                  noiseFunc, noiseFuncArgs, estimationFunc)
    
    drawGraphs(filepath)


main()



          


    

