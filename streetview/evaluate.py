import seaborn as sb
import numpy as np
import osmnx as ox
from geopy.distance import distance
from matplotlib import pyplot as plt


def evaluate_coverage_distance(df):
    sb.set_style("dark")
    f, axes = plt.subplots(2, 2, figsize=(12,8))
    axes[0][0].title.set_text(f"Coverage of [2011-2015]: {len(df)} / 5000 = {len(df)/5000*100:.2f}%")
    axes[0][1].title.set_text(f"Coverage of [2016-2020]: {len(df)} / 5000 = {len(df)/5000*100:.2f}%")
    sb.countplot(x="year_pre", data=df, ax=axes[0][0], palette=['#432371'])
    sb.countplot(x="year_post", data=df, ax=axes[0][1], palette=["#FAAE7B"])
    axes[0][0].set_xlabel('')
    axes[0][1].set_xlabel('')
    
    d1, i1 = zip(*get_closest_distances(df, 'pre'))
    d2, i2 = zip(*get_closest_distances(df, 'post'))
    sb.lineplot(x=range(len(d1)), y=d1, ax=axes[1][0])
    sb.lineplot(x=range(len(d2)), y=d2, ax=axes[1][1])
    axes[1][0].title.set_text(f"Top 50 closest distance of [2011-2015] panoramas")
    axes[1][1].title.set_text(f"Top 50 closest distance of [2016-2020] panoramas")
    return f

def get_closest_distances(df, suffix='pre', n=50):
    lat = df[f'lat_{suffix}'].values
    lon = df[f'lon_{suffix}'].values
    D = np.sqrt(np.square(lat[:,np.newaxis] - lat) + np.square(lon[:,np.newaxis] - lon))
    D = np.tril(D) + np.triu(np.ones_like(D))
    d = []
    for i in range(n):
        x, y = np.unravel_index(D.argmin(), D.shape)
        _d = distance((lat[x], lon[x]), (lat[y], lon[y])).m
        d.append((_d, x))
        D[x,:] = D[:,x] = 1
    return sorted(d)

def evaluate_spatial_distribution(df, city):
    sb.set_style("white")
    G = ox.graph_from_place(city, network_type='drive')
    try: 
        G = ox.simplify_graph(G)
    except: 
        G = G    
        
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,12))
    ox.plot.plot_graph(G, 
                       ax=ax1,
                       bgcolor='white', 
                       node_color='#696969',
                       edge_color="#A9A9A9",
                       edge_linewidth=0.8,
                       node_size=0,
                       save=False,
                       show=False)
    ax1.scatter(df.lon_anchor, df.lat_anchor, s=3, c='red', alpha=0.5)
    ax1.scatter(df.lon_pre, df.lat_pre, s=3, c='blue', alpha=0.5)
    
    ox.plot.plot_graph(G, 
                       ax=ax2,
                       bgcolor='white', 
                       node_color='#696969',
                       edge_color="#A9A9A9",
                       edge_linewidth=0.8,
                       node_size=0,
                       save=False,
                       show=False)
    ax2.scatter(df.lon_anchor, df.lat_anchor, s=3, c='red', alpha=0.5)
    ax2.scatter(df.lon_post, df.lat_post, s=3, c='blue', alpha=0.5)
    plt.show()