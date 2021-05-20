from matplotlib import pyplot as plt
import seaborn as sb
import osmnx as ox


def evaluate_spatial_distribution(df):
    sb.set_style("white")
    G = ox.graph_from_place('San Francisco, California, USA', 
                              network_type='drive')
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