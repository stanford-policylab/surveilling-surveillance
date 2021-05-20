from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import seaborn as sb 
import cv2 as cv

def plot_precision_recall():
    data = pd.read_csv("/home/haosheng/dataset/camera/test/test_result.csv")
    plt.figure(figsize=(8,6))
    sb.set_style("white")
    for f in [50, 200, 500, 1000]:
        data_plot = data.query(f"f == {f}")
        sb.lineplot(x="p", y="recall", 
                    data=data_plot, 
                    label=f"Pixel threshold: {f}",
                    linewidth=2.5,
                    ci=None)
    plt.xlim([0.145,1.05])
    plt.ylim([0,1.05])
    plt.axvline(x=0.583333, ymin=0, ymax=0.6, linestyle='-.', color='gray')
    plt.axhline(y=0.624400, xmin=0, xmax=0.48, linestyle='-.', color='gray')
    plt.plot(0.583333, 0.624400,'ro') 
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.legend()
    plt.savefig("figures/precision_recall.png")
