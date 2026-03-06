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
import numpy as np
import seaborn.objects as so

# Constants

IS_LOGGING = True
DATASETS_NAME = ["cifar10", "cifar100", "lfw"]

# Extra function
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

def main():
    datasetName = "cifar10"

    # ------------GAUSSIAN FILTER AND GAUSSIAN NOISE------------

    processFunc = GaussianFilterThreeChannelsImages
    processFuncArgs = {"filterSize": 3, "mode": "edge"}

    noiseFunc = addGaussianNoise
    noiseFuncArgs = {"a": 0, "sigma": 25}

    filepath = f"reports/test_{datasetName}_{processFunc.__name__}_{noiseFunc.__name__}.csv"

    try:
        x, y, cats = loadDataset(datasetName)
        data = []

        for idx, img in enumerate(x):
            noisy = noiseFunc(img, **noiseFuncArgs)

            start = time.time()
            procImg = processFunc(noisy, **processFuncArgs)
            period = (time.time() - start) * 1e3 # milliseconds

            # Estimations

            maeNoise = MAE(img, noisy)
            maeProc = MAE(img, procImg)
            maeImprov = maeNoise - maeProc

            psnrbb = PSNR_BB(img, noisy, procImg)

            snrNoise = SNR(img, noisy)
            snrProc = SNR(img, procImg)
            snrImprov = snrProc - snrNoise

            nrr = NRR(img, noisy, procImg)
            
            if cats is not None and y is not None and idx < len(y):
                cat = cats[int(y[idx])]
            else: cat = "unknown"

            data.append(dict(imageIndex=idx, 
                             category=cat, 
                             timeDelta=period,
                             
                             mae_noise=maeNoise,
                             mae_processed=maeProc,
                             mae_improvement=maeImprov,
                             psnr_bb=psnrbb,
                             snr_noise=snrNoise,
                             snr_processed=snrProc,
                             snr_improvement=snrImprov,
                             nrr=nrr))
            # LOGGING
            if (idx + 1) % 100 == 0:
                print(f"Succes: {(idx + 1)/len(x) * 100:.2f}%")

    except Exception as e:
        print(f"Error: {e}")

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    
    # Graph of NRR

    df["nrr_category"] = pd.cut(df["nrr"], 
                                bins=[-0.1, 0, 0.3, 0.6, 0.9, 1.1],
                                labels=["negative", "low", "average", "good", "excellent"])

    nrr_counts = df["nrr_category"].value_counts().reset_index()
    nrr_counts.columns = ["nrr_category", "count"]
    
    plot = (so.Plot(nrr_counts, x="nrr_category", y="count")
            .add(so.Bar())
            .label(x="NRR", y="Number"))
    plot.show()

    category_nrr = df.groupby("category")["nrr"].mean().reset_index()
    category_nrr = category_nrr.sort_values("nrr", ascending=False)
    
    plot = (so.Plot(category_nrr, x="category", y="nrr")
            .add(so.Bar())
            .label(x="Category", y="Average NRR"))
    plot.show()

    # GRAPH OF SNR

    

    # ------------MIDPOINT FILTER AND SALT & PEPPER NOISE------------

    processFunc = medianFilterThreeChannelsImages
    processFuncArgs = {"filterSize": 3, "mode": "edge"}

    noiseFunc = addSaltAndPepperNoise
    noiseFuncArgs = {"saltProb": 0.01, "pepperProb": 0.01}

    filepath = f"reports/test_{datasetName}_{processFunc.__name__}_{noiseFunc.__name__}.csv"

    try:
        x, y, cats = loadDataset(datasetName)
        data = []

        for idx, img in enumerate(x):
            noisy = noiseFunc(img, **noiseFuncArgs)

            start = time.time()
            procImg = processFunc(noisy, **processFuncArgs)
            period = (time.time() - start) * 1e3 # milliseconds

            # Estimations

            maeNoise = MAE(img, noisy)
            maeProc = MAE(img, procImg)
            maeImprov = maeNoise - maeProc

            psnrbb = PSNR_BB(img, noisy, procImg)

            snrNoise = SNR(img, noisy)
            snrProc = SNR(img, procImg)
            snrImprov = snrProc - snrNoise

            nrr = NRR(img, noisy, procImg)
            
            if cats is not None and y is not None and idx < len(y):
                cat = cats[int(y[idx])]
            else: cat = "unknown"

            data.append(dict(imageIndex=idx, 
                             category=cat, 
                             timeDelta=period,
                             
                             mae_noise=maeNoise,
                             mae_processed=maeProc,
                             mae_improvement=maeImprov,
                             psnr_bb=psnrbb,
                             snr_noise=snrNoise,
                             snr_processed=snrProc,
                             snr_improvement=snrImprov,
                             nrr=nrr))
            # LOGGING
            if (idx + 1) % 100 == 0:
                print(f"Succes: {(idx + 1)/len(x) * 100:.2f}%")

    except Exception as e:
        print(f"Error: {e}")

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    
    # Graph of NRR

    df["nrr_category"] = pd.cut(df["nrr"], 
                                bins=[-0.1, 0, 0.3, 0.6, 0.9, 1.1],
                                labels=["negative", "low", "average", "good", "excellent"])

    nrr_counts = df["nrr_category"].value_counts().reset_index()
    nrr_counts.columns = ["nrr_category", "count"]
    
    plot = (so.Plot(nrr_counts, x="nrr_category", y="count")
            .add(so.Bar())
            .label(x="NRR", y="Number"))
    plot.show()

    category_nrr = df.groupby("category")["nrr"].mean().reset_index()
    category_nrr = category_nrr.sort_values("nrr", ascending=False)
    
    plot = (so.Plot(category_nrr, x="category", y="nrr")
            .add(so.Bar())
            .label(x="Category", y="Average NRR"))
    plot.show()

    # GRAPH OF SNR

    
    # ------------ARITHMETIC MEAN FILTER AND UNIFORM NOISE------------

    processFunc = arithmeticMeanFilterThreeChannelsImages
    processFuncArgs = {"filterSize": 3, "mode": "edge"}

    noiseFunc = addUniformNoise
    noiseFuncArgs = {"beg": -50, "end": 50}

    filepath = f"reports/test_{datasetName}_{processFunc.__name__}_{noiseFunc.__name__}.csv"

    try:
        x, y, cats = loadDataset(datasetName)
        data = []

        for idx, img in enumerate(x):
            noisy = noiseFunc(img, **noiseFuncArgs)

            start = time.time()
            procImg = processFunc(noisy, **processFuncArgs)
            period = (time.time() - start) * 1e3 # milliseconds

            # Estimations

            maeNoise = MAE(img, noisy)
            maeProc = MAE(img, procImg)
            maeImprov = maeNoise - maeProc

            psnrbb = PSNR_BB(img, noisy, procImg)

            snrNoise = SNR(img, noisy)
            snrProc = SNR(img, procImg)
            snrImprov = snrProc - snrNoise

            nrr = NRR(img, noisy, procImg)
            
            if cats is not None and y is not None and idx < len(y):
                cat = cats[int(y[idx])]
            else: cat = "unknown"

            data.append(dict(imageIndex=idx, 
                             category=cat, 
                             timeDelta=period,
                             
                             mae_noise=maeNoise,
                             mae_processed=maeProc,
                             mae_improvement=maeImprov,
                             psnr_bb=psnrbb,
                             snr_noise=snrNoise,
                             snr_processed=snrProc,
                             snr_improvement=snrImprov,
                             nrr=nrr))
            # LOGGING
            if (idx + 1) % 100 == 0:
                print(f"Succes: {(idx + 1)/len(x) * 100:.2f}%")

    except Exception as e:
        print(f"Error: {e}")

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    
    # Graph of NRR

    df["nrr_category"] = pd.cut(df["nrr"], 
                                bins=[-0.1, 0, 0.3, 0.6, 0.9, 1.1],
                                labels=["negative", "low", "average", "good", "excellent"])

    nrr_counts = df["nrr_category"].value_counts().reset_index()
    nrr_counts.columns = ["nrr_category", "count"]
    
    plot = (so.Plot(nrr_counts, x="nrr_category", y="count")
            .add(so.Bar())
            .label(x="NRR", y="Number"))
    plot.show()

    category_nrr = df.groupby("category")["nrr"].mean().reset_index()
    category_nrr = category_nrr.sort_values("nrr", ascending=False)
    
    plot = (so.Plot(category_nrr, x="category", y="nrr")
            .add(so.Bar())
            .label(x="Category", y="Average NRR"))
    plot.show()

    # GRAPH OF SNR


main()