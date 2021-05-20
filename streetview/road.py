import os
import osmnx as ox
import pandas as pd
from tqdm import tqdm
import geopandas as gpd

from util import constants as C


def calculate_road_length():
    metas = []
    for name, place in tqdm(C.CITIES.items(), total=len(C.CITIES)):
        meta = ox.geocode_to_gdf(place)
        meta = meta.to_crs('EPSG:3395')
        meta['area'] = meta.geometry.apply(lambda x: x.area / 1e6)

        G = ox.graph_from_place(place, network_type='drive')
        try: 
            G = ox.simplify_graph(G)
        except: 
            G = G
        gdf = ox.utils_graph.graph_to_gdfs(G, nodes=False, edges=True)
        meta['length'] = gdf['length'].sum() / 1e3
        metas.append(meta)
    stats = gpd.GeoDataFrame(pd.concat(metas))[['display_name', 'area', 'length']] \
               .rename(columns={"area": "area(km^2)", "length": "length(km)"})
    print(stats)
    