import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from DIP.noise.reductions import *
from DIP.noise.generetors import *
from DIP.others.estimations import *
from DIP.others.translators import *

import keras
import time
import sklearn
import pandas as pd
import numpy as np
import seaborn.objects as so
import matplotlib.pyplot as plt

# Constants

IS_LOGGING = False
DATASETS_NAME = ["cifar10", "cifar100", "lfw"]

def loadDataset(datasetName): # Extra function for loading a dataset
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

    if IS_LOGGING: print("GAUSSIAN FILTER AND GAUSSIAN NOISE")

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

            mseNoise = MSE(img, noisy)
            mseProc = MSE(img, procImg)
            
            psnrNoise, _ = PSNR(img, noisy)
            psnrProc, _ = PSNR(img, procImg)

            maeNoise = MAE(img, noisy)
            maeProc = MAE(img, procImg)

            snrNoise = SNR(img, noisy)
            snrProc = SNR(img, procImg)

            nrr = NRR(img, noisy, procImg)
            
            if cats is not None and y is not None and idx < len(y):
                cat = cats[int(y[idx])]
            else: cat = "unknown"

            data.append(dict(image_index=idx, 
                             category=cat, 
                             time_delta=period,

                             mse_noise=mseNoise,
                             mse_processed=mseProc,

                             psnr_noise=psnrNoise,
                             psnr_processed=psnrProc,

                             mae_noise=maeNoise,
                             mae_processed=maeProc,
                             
                             snr_noise=snrNoise,
                             snr_processed=snrProc,
                             nrr=nrr))
            
            # LOGGING
            if (idx + 1) % 100 == 0 and IS_LOGGING:
                print(f"Succes: {(idx + 1)/len(x) * 100:.2f}%")

    except Exception as e:
        print(f"Error: {e}")

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

    # Graph of MSE

    data = df

    data["mse_improvement"] = data["mse_noise"]-data["mse_processed"]
    data["type_change_result"] = data["mse_improvement"].apply(lambda x: "negative" if x<=0 else "positive")
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Number Of Images By Types Of Change Result According to MSE"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["mse_improvement"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="mse_improvement")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average MSE Improvement", title="Average MSE Improvement By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # Graph of PSNR

    data = df

    data["psnr_improvement"] = data["psnr_processed"]-data["psnr_noise"]
    data["type_change_result"] = data["psnr_improvement"].apply(lambda x: "negative" if x<=0 else "positive")
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Number Of Images By Types Of Change Result According to PSNR"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["psnr_improvement"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="psnr_improvement")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average PSNR Improvement", title="Average PSNR Improvement By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # Graph of NRR

    data = df

    data["type_change_result"] = pd.cut(data["nrr"], 
                                        bins=[-1e-8, 0, 3e-1, 6e-1, 9e-1, 11e-1],
                                        labels=["Negative", "Low", "Average", "Good", "Excellent"])
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Distribution Of Images According To NRR"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["nrr"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="nrr")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average NRR Estimation", title="Average NRR Estimation By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # GRAPH OF SNR

    data = df

    data["snr_improvement"] = data["snr_processed"]-data["snr_noise"]
    data["type_change_result"] = data["snr_improvement"].apply(lambda x: "negative" if x<=0 else "positive")
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Number Of Images By Types Of Change Result According to NSR"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["snr_improvement"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="snr_improvement")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average SNR Improvement", title="Average SNR Improvement By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # Graph of MAE

    data = df

    data["mae_improvement"] = data["mae_noise"]-data["mae_processed"]
    data["type_change_result"] = data["mae_improvement"].apply(lambda x: "negative" if x<=0 else "positive")
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Number Of Images By Types Of Change Result According to MAE"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["mae_improvement"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="mae_improvement")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average MAE Improvement", title="Average MAE Improvement By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # ----------- MEDIAN FILTER AND SALT & PEPPER NOISE -----------

    if IS_LOGGING: print("MEDIAN FILTER AND SALT & PEPPER NOISE")

    processFunc = medianFilterThreeChannelsImages
    processFuncArgs = {"filterSize": 3, "mode": "edge"}

    noiseFunc = addSaltAndPepperNoise
    noiseFuncArgs = {"saltProb": 0.05, "pepperProb": 0.05}

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

            mseNoise = MSE(img, noisy)
            mseProc = MSE(img, procImg)
            
            psnrNoise, _ = PSNR(img, noisy)
            psnrProc, _ = PSNR(img, procImg)

            maeNoise = MAE(img, noisy)
            maeProc = MAE(img, procImg)

            snrNoise = SNR(img, noisy)
            snrProc = SNR(img, procImg)

            nrr = NRR(img, noisy, procImg)
            
            if cats is not None and y is not None and idx < len(y):
                cat = cats[int(y[idx])]
            else: cat = "unknown"

            data.append(dict(image_index=idx, 
                             category=cat, 
                             time_delta=period,

                             mse_noise=mseNoise,
                             mse_processed=mseProc,

                             psnr_noise=psnrNoise,
                             psnr_processed=psnrProc,

                             mae_noise=maeNoise,
                             mae_processed=maeProc,
                             
                             snr_noise=snrNoise,
                             snr_processed=snrProc,
                             nrr=nrr))
            
            # LOGGING
            if (idx + 1) % 100 == 0 and IS_LOGGING:
                print(f"Succes: {(idx + 1)/len(x) * 100:.2f}%")

    except Exception as e:
        print(f"Error: {e}")

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

    # Graph of MSE

    data = df

    data["mse_improvement"] = data["mse_noise"]-data["mse_processed"]
    data["type_change_result"] = data["mse_improvement"].apply(lambda x: "negative" if x<=0 else "positive")
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Number Of Images By Types Of Change Result According to MSE"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["mse_improvement"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="mse_improvement")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average MSE Improvement", title="Average MSE Improvement By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # Graph of PSNR

    data = df

    data["psnr_improvement"] = data["psnr_processed"]-data["psnr_noise"]
    data["type_change_result"] = data["psnr_improvement"].apply(lambda x: "negative" if x<=0 else "positive")
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Number Of Images By Types Of Change Result According to PSNR"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["psnr_improvement"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="psnr_improvement")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average PSNR Improvement", title="Average PSNR Improvement By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # Graph of NRR

    data = df

    data["type_change_result"] = pd.cut(data["nrr"], 
                                        bins=[-1e-8, 0, 3e-1, 6e-1, 9e-1, 11e-1],
                                        labels=["Negative", "Low", "Average", "Good", "Excellent"])
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Distribution Of Images According To NRR"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["nrr"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="nrr")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average NRR Estimation", title="Average NRR Estimation By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # GRAPH OF SNR

    data = df

    data["snr_improvement"] = data["snr_processed"]-data["snr_noise"]
    data["type_change_result"] = data["snr_improvement"].apply(lambda x: "negative" if x<=0 else "positive")
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Number Of Images By Types Of Change Result According to NSR"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["snr_improvement"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="snr_improvement")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average SNR Improvement", title="Average SNR Improvement By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # Graph of MAE

    data = df

    data["mae_improvement"] = data["mae_noise"]-data["mae_processed"]
    data["type_change_result"] = data["mae_improvement"].apply(lambda x: "negative" if x<=0 else "positive")
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Number Of Images By Types Of Change Result According to MAE"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["mae_improvement"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="mae_improvement")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average MAE Improvement", title="Average MAE Improvement By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()
    
    # ----------- MIDPOINT FILER AND UNIFORM NOISE -----------
    
    if IS_LOGGING: print("MIDPOINT FILER AND UNIFORM NOISE")

    processFunc = midpointFilterThreeChannelsImages
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

            mseNoise = MSE(img, noisy)
            mseProc = MSE(img, procImg)
            
            psnrNoise, _ = PSNR(img, noisy)
            psnrProc, _ = PSNR(img, procImg)

            maeNoise = MAE(img, noisy)
            maeProc = MAE(img, procImg)

            snrNoise = SNR(img, noisy)
            snrProc = SNR(img, procImg)

            nrr = NRR(img, noisy, procImg)
            
            if cats is not None and y is not None and idx < len(y):
                cat = cats[int(y[idx])]
            else: cat = "unknown"

            data.append(dict(image_index=idx, 
                             category=cat, 
                             time_delta=period,

                             mse_noise=mseNoise,
                             mse_processed=mseProc,

                             psnr_noise=psnrNoise,
                             psnr_processed=psnrProc,

                             mae_noise=maeNoise,
                             mae_processed=maeProc,
                             
                             snr_noise=snrNoise,
                             snr_processed=snrProc,
                             nrr=nrr))
            
            # LOGGING
            if (idx + 1) % 100 == 0 and IS_LOGGING:
                print(f"Succes: {(idx + 1)/len(x) * 100:.2f}%")

    except Exception as e:
        print(f"Error: {e}")

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

    # Graph of MSE

    data = df

    data["mse_improvement"] = data["mse_noise"]-data["mse_processed"]
    data["type_change_result"] = data["mse_improvement"].apply(lambda x: "negative" if x<=0 else "positive")
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Number Of Images By Types Of Change Result According to MSE"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["mse_improvement"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="mse_improvement")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average MSE Improvement", title="Average MSE Improvement By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # Graph of PSNR

    data = df

    data["psnr_improvement"] = data["psnr_processed"]-data["psnr_noise"]
    data["type_change_result"] = data["psnr_improvement"].apply(lambda x: "negative" if x<=0 else "positive")
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Number Of Images By Types Of Change Result According to PSNR"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["psnr_improvement"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="psnr_improvement")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average PSNR Improvement", title="Average PSNR Improvement By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # Graph of NRR

    data = df

    data["type_change_result"] = pd.cut(data["nrr"], 
                                        bins=[-1e-8, 0, 3e-1, 6e-1, 9e-1, 11e-1],
                                        labels=["Negative", "Low", "Average", "Good", "Excellent"])
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Distribution Of Images According To NRR"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["nrr"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="nrr")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average NRR Estimation", title="Average NRR Estimation By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # GRAPH OF SNR

    data = df

    data["snr_improvement"] = data["snr_processed"]-data["snr_noise"]
    data["type_change_result"] = data["snr_improvement"].apply(lambda x: "negative" if x<=0 else "positive")
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Number Of Images By Types Of Change Result According to NSR"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["snr_improvement"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="snr_improvement")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average SNR Improvement", title="Average SNR Improvement By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # Graph of MAE

    data = df

    data["mae_improvement"] = data["mae_noise"]-data["mae_processed"]
    data["type_change_result"] = data["mae_improvement"].apply(lambda x: "negative" if x<=0 else "positive")
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Number Of Images By Types Of Change Result According to MAE"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["mae_improvement"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="mae_improvement")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average MAE Improvement", title="Average MAE Improvement By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # ----------- ARITHMETICMEAN FILTER AND SALT & PEPPER NOISE -----------

    if IS_LOGGING: print("ARITHMETICMEAN FILTER AND SALT & PEPPER NOISE")

    processFunc = arithmeticMeanFilterThreeChannelsImages
    processFuncArgs = {"filterSize": 3, "mode": "edge"}

    noiseFunc = addSaltAndPepperNoise
    noiseFuncArgs = {"saltProb": 0.05, "pepperProb": 0.05}

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

            mseNoise = MSE(img, noisy)
            mseProc = MSE(img, procImg)
            
            psnrNoise, _ = PSNR(img, noisy)
            psnrProc, _ = PSNR(img, procImg)

            maeNoise = MAE(img, noisy)
            maeProc = MAE(img, procImg)

            snrNoise = SNR(img, noisy)
            snrProc = SNR(img, procImg)

            nrr = NRR(img, noisy, procImg)
            
            if cats is not None and y is not None and idx < len(y):
                cat = cats[int(y[idx])]
            else: cat = "unknown"

            data.append(dict(image_index=idx, 
                             category=cat, 
                             time_delta=period,

                             mse_noise=mseNoise,
                             mse_processed=mseProc,

                             psnr_noise=psnrNoise,
                             psnr_processed=psnrProc,

                             mae_noise=maeNoise,
                             mae_processed=maeProc,
                             
                             snr_noise=snrNoise,
                             snr_processed=snrProc,
                             nrr=nrr))
            
            # LOGGING
            if (idx + 1) % 100 == 0 and IS_LOGGING:
                print(f"Succes: {(idx + 1)/len(x) * 100:.2f}%")

    except Exception as e:
        print(f"Error: {e}")

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

    # Graph of MSE

    data = df

    data["mse_improvement"] = data["mse_noise"]-data["mse_processed"]
    data["type_change_result"] = data["mse_improvement"].apply(lambda x: "negative" if x<=0 else "positive")
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Number Of Images By Types Of Change Result According to MSE"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["mse_improvement"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="mse_improvement")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average MSE Improvement", title="Average MSE Improvement By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # Graph of PSNR

    data = df

    data["psnr_improvement"] = data["psnr_processed"]-data["psnr_noise"]
    data["type_change_result"] = data["psnr_improvement"].apply(lambda x: "negative" if x<=0 else "positive")
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Number Of Images By Types Of Change Result According to PSNR"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["psnr_improvement"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="psnr_improvement")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average PSNR Improvement", title="Average PSNR Improvement By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # Graph of NRR

    data = df

    data["type_change_result"] = pd.cut(data["nrr"], 
                                        bins=[-1e-8, 0, 3e-1, 6e-1, 9e-1, 11e-1],
                                        labels=["Negative", "Low", "Average", "Good", "Excellent"])
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Distribution Of Images According To NRR"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["nrr"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="nrr")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average NRR Estimation", title="Average NRR Estimation By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # GRAPH OF SNR

    data = df

    data["snr_improvement"] = data["snr_processed"]-data["snr_noise"]
    data["type_change_result"] = data["snr_improvement"].apply(lambda x: "negative" if x<=0 else "positive")
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Number Of Images By Types Of Change Result According to NSR"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["snr_improvement"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="snr_improvement")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average SNR Improvement", title="Average SNR Improvement By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

    # Graph of MAE

    data = df

    data["mae_improvement"] = data["mae_noise"]-data["mae_processed"]
    data["type_change_result"] = data["mae_improvement"].apply(lambda x: "negative" if x<=0 else "positive")
    plot = (so.Plot(data, x="type_change_result")
            .add(so.Bar(), so.Count())
            .label(x="Change Result", y="Number Of Images", title="Number Of Images By Types Of Change Result According to MAE"))
    plot.show()

    _, ax = plt.subplots(figsize=(10, 6))
    data = data.groupby("category")["mae_improvement"].mean().reset_index()
    plot = (so.Plot(data, x="category", y="mae_improvement")
            .add(so.Bar(color="green"))
            .on(ax)
            .label(x="Image Category", y="Average MAE Improvement", title="Average MAE Improvement By Categories"))
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plot.show()

main()