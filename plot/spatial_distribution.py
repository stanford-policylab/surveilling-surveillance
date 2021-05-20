from util import constants as C
import osmnx as ox
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sb
sb.set()


def plot_samples(
        meta_file_path="/home/haosheng/dataset/camera/deployment/verified_0425.csv"):
    data = pd.read_csv(meta_file_path)
    for city, place in list(C.CITIES.items()):
        with open(f"/home/haosheng/dataset/camera/shape/graph/{city}.pkl", "rb") as f:
            G = pkl.load(f)
        ox.plot.plot_graph(G,
                           figsize=(12, 12),
                           bgcolor='white',
                           node_color='#696969',
                           edge_color="#A9A9A9",
                           edge_linewidth=0.8,
                           node_size=0,
                           edge_alpha=0.5,
                           save=False,
                           show=False)
        sample = data.query(f'city == "{city}"')

        plt.scatter(
            sample.lon_anchor,
            sample.lat_anchor,
            s=0.2,
            c='blue',
            alpha=1)
        plt.tight_layout()
        plt.savefig(f"figures/samples_{city}.png")
        print(f"Save figure to [figures/samples_{city}.png]")


def plot_prepost(
        meta_file_path="/home/haosheng/dataset/camera/deployment/verified_prepost_0425.csv"):
    data = pd.read_csv(meta_file_path)

    for city, place in list(C.CITIES.items())[:10]:
        with open(f"/home/haosheng/dataset/camera/shape/graph/{city}.pkl", "rb") as f:
            G = pkl.load(f)
        ox.plot.plot_graph(G,
                           figsize=(12, 12),
                           bgcolor='white',
                           node_color='#696969',
                           edge_color="#A9A9A9",
                           edge_linewidth=0.8,
                           node_size=0,
                           edge_alpha=0.5,
                           save=False,
                           show=False)

        print("Generating the plot .. ")

        pre = data.query(
            f'camera_count > 0 and split == "pre" and city == "{city}"')
        post = data.query(
            f'camera_count > 0 and split == "post" and city == "{city}"')

        plt.scatter(
            pre.lon_anchor,
            pre.lat_anchor,
            s=150,
            facecolors='none',
            edgecolors='red',
            linewidth=2.0,
            marker='o')
        plt.scatter(
            post.lon_anchor,
            post.lat_anchor,
            s=120,
            c='black',
            marker='x')
        plt.tight_layout()
        plt.savefig(f"figures/prepost_spatial_distribution_{city}.png")
        print(
            f"Save figure to [figures/prepost_spatial_distribution_{city}.png]")


def plot_post(
        meta_file_path="/home/haosheng/dataset/camera/deployment/verified_0425.csv"):
    data = pd.read_csv(meta_file_path)
    for city, place in C.CITIES.items():
        with open(f"/home/haosheng/dataset/camera/shape/graph/{city}.pkl", "rb") as f:
            G = pkl.load(f)

        ox.plot.plot_graph(G,
                           figsize=(12, 12),
                           bgcolor='white',
                           node_color='#696969',
                           edge_color="#A9A9A9",
                           edge_linewidth=0.8,
                           node_size=0,
                           edge_alpha=0.5,
                           save=False,
                           show=False)

        print("Generating the plot .. ")

        pre = data.query(f'camera_count > 0 and city == "{city}"')
        post = data.query(f'camera_count > 0 and city == "{city}"')

        plt.scatter(
            pre.lon_anchor,
            pre.lat_anchor,
            color='red',
            #color='#BE0000',
            s=30,
            linewidth=2.0,
            marker='o',
            alpha=1)
        plt.tight_layout()
        plt.savefig(f"figures/post_spatial_distribution_{city}.png")
        print(f"Save figure to [figures/post_spatial_distribution_{city}.png]")


def plot_spatial_distribution():
    plot_samples()
    plot_prepost()
    plot_post()
