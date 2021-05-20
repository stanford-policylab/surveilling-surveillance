import pandas as pd
from matplotlib import pyplot as plt
from util import constants as C
from scipy.stats.mstats import winsorize
import numpy as np
import matplotlib.collections as collections
import seaborn as sb
import matplotlib

LABEL = [('SF', 'San Francisco, California, USA'), ('Chicago',
                                                    'Chicago, Illinois, USA'), ('NYC', 'New York City, New York, USA')]


def plot_coverage():
    plt.figure(figsize=(8, 4))
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 15}

    matplotlib.rc('font', **font)
    T = 60
    COLOR = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    for i, (name, place) in enumerate(LABEL):
        data = pd.read_csv(
            f"/home/haosheng/dataset/camera/sample/meta_0228/{name}_coverage.csv")
        sb.kdeplot(data.coverage, label=place.split(",")[0], linewidth=2)
        threshold = np.clip(data.coverage, 0, T).mean()
        plt.axvline(x=threshold, linestyle='-.', color=COLOR[i])
        print(f"Average coverage for city {place}: {threshold}")
    plt.xlim([0, 120])

    plt.legend(loc='upper right')
    plt.xlabel("Estimated Road Segment Coverage (meter)")
    plt.ylabel("Probability Density")

    t = np.arange(T, 130, 0.01)
    collection = collections.BrokenBarHCollection.span_where(
        t, ymin=0, ymax=1, where=t > 0, facecolor='gray', alpha=0.15)
    ax = plt.gca()
    ax.add_collection(collection)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("figures/coverage.png")
