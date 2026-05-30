import cv2
import numpy as np
import keras
import sklearn
import pandas as pd
import seaborn.objects as so
import matplotlib.pyplot as plt
import os

from DIP.others.estimations import showHystogramThreeChannelsImages
from DIP.others.translators import linearTransformation, gammaCorrection, logarithmicTransformation, histogramEqualizationThreeChannelsImages, claheThreeChannelsImages
from DIP.noise.generators import addGaussianNoise, addSaltAndPepperNoise, addUniformNoise, addPoissonNoise
from DIP.others.estimations import MSE, PSNR, MAE, NRR, SSIM, SNR
from DIP.noise.reductions import GaussianFilterThreeChannelsImages, geometricMeanFilterThreeChannelsImages, arithmeticMeanFilterThreeChannelsImages, medianFilterThreeChannelsImages, bilateralFilterThreeChannelImages

IS_LOGGING = True

def loadDataset(datasetName):
    match(datasetName):
        case "cifar10":
            (_, _), (x, y) = keras.datasets.cifar10.load_data()
            cats = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]
        case "cifar100":
            (_, _), (x, y) = keras.datasets.cifar100.load_data(label_mode="fine")
            cats = ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", 
                    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", 
                    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", 
                    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", 
                    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", 
                    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
                    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
                    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
                    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
                    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
                    "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
                    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
                    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
                    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
                    "worm"]
        case "lfw":
            lfw = sklearn.datasets.fetch_lfw_people(min_faces_per_person=20, resize=0.5)
            x = (lfw.images * 255).astype(np.uint8)
            x = np.stack([x, x, x], axis=-1)
            y = lfw.target
            cats = lfw.target_names
        case _:
            raise ValueError("Wrong the dataset name")

    return x, y, cats

def main():
    """
    # --------Linear and nonlinear transformations--------

    filepaths = ["imgs/person.jpg", "imgs/dog.jpg",
                 "imgs/text.jpg", "imgs/houses.jpg",
                 "imgs/rose.jpg", "imgs/flowers.jpg",
                 "imgs/reflection.jpg", "imgs/laptop.jpg"]
    
    for filepath in filepaths:
        img = cv2.imread(filepath)

        if img is None:
            print("FileNotFound Exception")
            return

        if IS_LOGGING: print("Source Image")
        showHystogramThreeChannelsImages(img, title="Source Image: " + filepath, isNormalized=False)

        if IS_LOGGING: print("Linear Transformation (alpha=1.5, beta=10)")
        res = linearTransformation(img, alpha=1.5,
                                   beta=10)
        showHystogramThreeChannelsImages(res, title="Linear Transformation (alpha=1.5, beta=10): " + filepath, isNormalized=False)

        if IS_LOGGING: print("Gamma Correction (gamma=0.5)")
        res = gammaCorrection(img, gamma=0.5)
        showHystogramThreeChannelsImages(res, title="Gamma Correction (gamma=0.5):" + filepath, isNormalized=False)

        if IS_LOGGING: print("Gamma Correction (gamma=3)")
        res = gammaCorrection(img, gamma=3)
        showHystogramThreeChannelsImages(res, title="Gamma Correction (gamma=3): " + filepath, isNormalized=False)

        if IS_LOGGING: print("Logarithmic Transformation (c=25)")
        res = logarithmicTransformation(img, c=25)
        showHystogramThreeChannelsImages(res, title="Logarithmic Transformation (c=25): " + filepath, isNormalized=False)

        if IS_LOGGING: print("Histogram Equalization")
        res = histogramEqualizationThreeChannelsImages(img)
        showHystogramThreeChannelsImages(res, title="Histogram Equalization: " + filepath, isNormalized=False)

        if IS_LOGGING: print("CLAHE")
        res = claheThreeChannelsImages(img)
        showHystogramThreeChannelsImages(res, title="CLAHE: " + filepath, isNormalized=False)
    # --------GAUSSIAN NOISE--------

    if IS_LOGGING: print("GAUSSIAN NOISE")

    datasets = ["cifar10", "cifar100",
                "lfw"]
    
    func_arg = [(GaussianFilterThreeChannelsImages, {"filterSize": 3, "mode": "edge", "sigma": 0.8}),
                (arithmeticMeanFilterThreeChannelsImages, {"filterSize": 3, "mode": "edge"}),
                (geometricMeanFilterThreeChannelsImages, {"filterSize": 3, "mode": "edge"}),
                (medianFilterThreeChannelsImages, {"filterSize": 3, "mode": "edge"}),
                (bilateralFilterThreeChannelImages, {"d": 3, "sigmaColor": 100, "sigmaSpace": 35, "mode": "edge"})]
    
    for dataset in datasets:
        x, y, cats = loadDataset(dataset)
        data = []
        for func, arg in func_arg:
            for idx, img in enumerate(x):
                noisyImg = addGaussianNoise(img=img, a=0,
                                            sigma=25)
                procImg = func(noisyImg, **arg)

                mse = MSE(img, procImg)
                psnr, _ = PSNR(img, procImg)
                mae = MAE(img, procImg)
                snr = SNR(img, procImg)
                nrr = NRR(img, noisyImg, procImg)
                ssim = SSIM(img, procImg)

                cat = cats[int(y[idx])]

                data.append(dict(image_index=idx, category=cat, 
                                 mse=mse, psnr=psnr,
                                 mae=mae, snr=snr,
                                 nrr=nrr, ssim=ssim))
                
                if (idx + 1) % 100 == 0 and IS_LOGGING:
                    print(f"Dataset: {dataset} -> Success: {(idx + 1)/len(x) * 100:.2f}%")

            filepath = f"reports/{dataset}_{func.__name__}_{addGaussianNoise.__name__}.csv"
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["mse"].mean().reset_index(), x="category", y="mse")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean MSE",
                        title="Mean MSE by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["psnr"].mean().reset_index(), x="category", y="psnr")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean PSNR",
                        title="Mean PSNR by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["mae"].mean().reset_index(), x="category", y="mae")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean MAE",
                        title="Mean MAE by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["snr"].mean().reset_index(), x="category", y="snr")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean SNR",
                        title="Mean SNR by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["nrr"].mean().reset_index(), x="category", y="nrr")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean NRR",
                        title="Mean NRR by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["ssim"].mean().reset_index(), x="category", y="ssim")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean SSIM",
                        title="Mean SSIM by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

    # --------SALT & PEPPER--------

    if IS_LOGGING: print("SALT & PEPPER")
    
    datasets = ["cifar10", "cifar100",
                "lfw"]
    
    func_arg = [(GaussianFilterThreeChannelsImages, {"filterSize": 3, "mode": "edge", "sigma": 0.8}),
                (arithmeticMeanFilterThreeChannelsImages, {"filterSize": 3, "mode": "edge"}),
                (geometricMeanFilterThreeChannelsImages, {"filterSize": 3, "mode": "edge"}),
                (medianFilterThreeChannelsImages, {"filterSize": 3, "mode": "edge"}),
                (bilateralFilterThreeChannelImages, {"d": 3, "sigmaColor": 100, "sigmaSpace": 35, "mode": "edge"})]
    
    for dataset in datasets:
        x, y, cats = loadDataset(dataset)
        data = []
        for func, arg in func_arg:
            for idx, img in enumerate(x):
                noisyImg = addSaltAndPepperNoise(img=img, saltProb=0.03,
                                                 pepperProb=0.03)
                procImg = func(noisyImg, **arg)

                mse = MSE(img, procImg)
                psnr, _ = PSNR(img, procImg)
                mae = MAE(img, procImg)
                snr = SNR(img, procImg)
                nrr = NRR(img, noisyImg, procImg)
                ssim = SSIM(img, procImg)

                cat = cats[int(y[idx])]

                data.append(dict(image_index=idx, category=cat, 
                                 mse=mse, psnr=psnr,
                                 mae=mae, snr=snr,
                                 nrr=nrr, ssim=ssim))
                
                if (idx + 1) % 100 == 0 and IS_LOGGING:
                    print(f"Dataset: {dataset} -> Success: {(idx + 1)/len(x) * 100:.2f}%")

            filepath = f"reports/{dataset}_{func.__name__}_{addGaussianNoise.__name__}.csv"
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["mse"].mean().reset_index(), x="category", y="mse")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean MSE",
                        title="Mean MSE by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["psnr"].mean().reset_index(), x="category", y="psnr")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean PSNR",
                        title="Mean PSNR by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["mae"].mean().reset_index(), x="category", y="mae")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean MAE",
                        title="Mean MAE by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["snr"].mean().reset_index(), x="category", y="snr")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean SNR",
                        title="Mean SNR by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["nrr"].mean().reset_index(), x="category", y="nrr")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean NRR",
                        title="Mean NRR by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["ssim"].mean().reset_index(), x="category", y="ssim")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean SSIM",
                        title="Mean SSIM by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

    # --------UNIFORM NOISE--------

    if IS_LOGGING: print("UNIFORM NOISE")

    datasets = ["cifar10", "cifar100",
                "lfw"]
    
    func_arg = [(GaussianFilterThreeChannelsImages, {"filterSize": 3, "mode": "edge", "sigma": 0.8}),
                (arithmeticMeanFilterThreeChannelsImages, {"filterSize": 3, "mode": "edge"}),
                (geometricMeanFilterThreeChannelsImages, {"filterSize": 3, "mode": "edge"}),
                (medianFilterThreeChannelsImages, {"filterSize": 3, "mode": "edge"}),
                (bilateralFilterThreeChannelImages, {"d": 3, "sigmaColor": 100, "sigmaSpace": 35, "mode": "edge"})]
    
    for dataset in datasets:
        x, y, cats = loadDataset(dataset)
        data = []
        for func, arg in func_arg:
            for idx, img in enumerate(x):
                noisyImg = addUniformNoise(img=img, beg=-35,
                                           end=35)
                procImg = func(noisyImg, **arg)

                mse = MSE(img, procImg)
                psnr, _ = PSNR(img, procImg)
                mae = MAE(img, procImg)
                snr = SNR(img, procImg)
                nrr = NRR(img, noisyImg, procImg)
                ssim = SSIM(img, procImg)

                cat = cats[int(y[idx])]

                data.append(dict(image_index=idx, category=cat, 
                                 mse=mse, psnr=psnr,
                                 mae=mae, snr=snr,
                                 nrr=nrr, ssim=ssim))
                
                if (idx + 1) % 100 == 0 and IS_LOGGING:
                    print(f"Dataset: {dataset} -> Success: {(idx + 1)/len(x) * 100:.2f}%")

            filepath = f"reports/{dataset}_{func.__name__}_{addGaussianNoise.__name__}.csv"
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["mse"].mean().reset_index(), x="category", y="mse")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean MSE",
                        title="Mean MSE by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["psnr"].mean().reset_index(), x="category", y="psnr")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean PSNR",
                        title="Mean PSNR by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["mae"].mean().reset_index(), x="category", y="mae")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean MAE",
                        title="Mean MAE by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["snr"].mean().reset_index(), x="category", y="snr")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean SNR",
                        title="Mean SNR by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["nrr"].mean().reset_index(), x="category", y="nrr")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean NRR",
                        title="Mean NRR by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["ssim"].mean().reset_index(), x="category", y="ssim")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean SSIM",
                        title="Mean SSIM by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

    # --------POISSON NOISE--------

    if IS_LOGGING: print("POISSON NOISE")

    datasets = ["cifar10", "cifar100",
                "lfw"]
    
    func_arg = [(GaussianFilterThreeChannelsImages, {"filterSize": 3, "mode": "edge", "sigma": 0.8}),
                (arithmeticMeanFilterThreeChannelsImages, {"filterSize": 3, "mode": "edge"}),
                (geometricMeanFilterThreeChannelsImages, {"filterSize": 3, "mode": "edge"}),
                (medianFilterThreeChannelsImages, {"filterSize": 3, "mode": "edge"}),
                (bilateralFilterThreeChannelImages, {"d": 3, "sigmaColor": 100, "sigmaSpace": 35, "mode": "edge"})]
    
    for dataset in datasets:
        x, y, cats = loadDataset(dataset)
        data = []
        for func, arg in func_arg:
            for idx, img in enumerate(x):
                noisyImg = addPoissonNoise(img=img, scale=35)
                procImg = func(noisyImg, **arg)

                mse = MSE(img, procImg)
                psnr, _ = PSNR(img, procImg)
                mae = MAE(img, procImg)
                snr = SNR(img, procImg)
                nrr = NRR(img, noisyImg, procImg)
                ssim = SSIM(img, procImg)

                cat = cats[int(y[idx])]

                data.append(dict(image_index=idx, category=cat, 
                                 mse=mse, psnr=psnr,
                                 mae=mae, snr=snr,
                                 nrr=nrr, ssim=ssim))
                
                if (idx + 1) % 100 == 0 and IS_LOGGING:
                    print(f"Dataset: {dataset} -> Success: {(idx + 1)/len(x) * 100:.2f}%")

            filepath = f"reports/{dataset}_{func.__name__}_{addGaussianNoise.__name__}.csv"
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["mse"].mean().reset_index(), x="category", y="mse")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean MSE",
                        title="Mean MSE by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["psnr"].mean().reset_index(), x="category", y="psnr")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean PSNR",
                        title="Mean PSNR by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["mae"].mean().reset_index(), x="category", y="mae")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean MAE",
                        title="Mean MAE by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["snr"].mean().reset_index(), x="category", y="snr")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean SNR",
                        title="Mean SNR by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["nrr"].mean().reset_index(), x="category", y="nrr")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean NRR",
                        title="Mean NRR by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

            _, ax = plt.subplots(figsize=(10, 6))
            plot = (so.Plot(df.groupby("category")["ssim"].mean().reset_index(), x="category", y="ssim")
                    .add(so.Bar())
                    .on(ax)
                    .label(x="Category", y="Mean SSIM",
                        title="Mean SSIM by Category"))
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            plot.show()

    if IS_LOGGING: print("VISUALIZATION ON PERSONAL IMAGES")
    """
    
    filepaths = ["imgs/person.jpg", "imgs/dog.jpg",
                 "imgs/text.jpg", "imgs/houses.jpg",
                 "imgs/rose.jpg", "imgs/flowers.jpg",
                 "imgs/reflection.jpg", "imgs/laptop.jpg"]
    
    best_filters = {"Gaussian": ("Gaussian", lambda img: GaussianFilterThreeChannelsImages(img, filterSize=7, sigma=2.0)),
                    "Uniform": ("Gaussian", lambda img: GaussianFilterThreeChannelsImages(img, filterSize=7, sigma=2.0)),
                    "Salt & Pepper": ("Median", lambda img: medianFilterThreeChannelsImages(img, filterSize=5)),
                    "Poisson": ("Bilateral", lambda img: bilateralFilterThreeChannelImages(img, d=7, sigmaColor=200, sigmaSpace=100))}
    
    noise_funcs = {"Gaussian": lambda img: addGaussianNoise(img, a=5, sigma=10),
                   "Uniform": lambda img: addUniformNoise(img, beg=-50, end=50),
                   "Salt & Pepper": lambda img: addSaltAndPepperNoise(img, saltProb=0.1, pepperProb=0.1),
                   "Poisson": lambda img: addPoissonNoise(img, scale=30)}
    
    for filepath in filepaths:
        img = cv2.imread(filepath)
        if img is None:
            print(f"FileNotFound Exception: {filepath}")
            continue
        
        for noise_name, noise_func in noise_funcs.items():
            noisy_img = noise_func(img)
            filter_name, filter_func = best_filters[noise_name]
            restored_img = filter_func(noisy_img)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"{os.path.basename(filepath)}", fontsize=14)
            
            axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Source")
            axes[0].axis("off")
            
            axes[1].imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f"Noisy ({noise_name})")
            axes[1].axis("off")
            
            axes[2].imshow(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
            axes[2].set_title(f"Processed ({filter_name})")
            axes[2].axis("off")
            
            plt.tight_layout()
            plt.savefig(f"reports/vis_{os.path.basename(filepath).split('.')[0]}_{noise_name}.png", dpi=150, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    main()