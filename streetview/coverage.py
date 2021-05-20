import os
import osmnx as ox
from shapely.geometry import Point
from shapely.ops import nearest_points
from geopy import distance
from tqdm import tqdm
import pickle as pkl
import pandas as pd
import geopandas as gpd
import multiprocessing
import numpy as np

from util import constants as C


def get_buildings(city, city_tag):
    tags = tags = {'building': True}
    building_path = f"/share/data/camera/shape/building/{city}.pkl"
    if False:#os.path.exists(building_path):
        with open(building_path, "rb") as f:
            gdf = pkl.load(f)
    else:
        gdf = ox.geometries_from_place(city_tag, tags)
        with open(building_path, "wb") as f:
            pkl.dump(gdf, f) 
    rows = []
    for rid, row in tqdm(gdf.iterrows(), total=len(gdf)):
        if isinstance(row['geometry'], Point):
                continue
        row['centroid_lat'] = row['geometry'].centroid.y
        row['centroid_lon'] = row['geometry'].centroid.x
        rows.append(row)
    buildings = gpd.GeoDataFrame(rows)
    return buildings

def get_coverage(lat, lon, buildings, t=0.005, default=50):
    dist = default
    try:
        near_buildings = buildings.query(f"{lat-t} < centroid_lat < {lat+t} and \
                                           {lon-t} < centroid_lon < {lon+t}")
        for rid, row in near_buildings.iterrows():
            building = row['geometry']

            p = nearest_points(building, Point(lon, lat))[0]
            _lat, _lon = p.y, p.x
            _dist = distance.distance((lat, lon), (_lat, _lon)).m
            dist = min(dist, _dist)
            
    except Exception as e:
        print(str(e))
        pass
    return 2 * dist

def get_coverage_df(rtuple):
    global buildings
    rid, row = rtuple
    lat, lon = row['lat'], row['lon']
    row['coverage'] = get_coverage(lat, lon, buildings)
    return row

def calculate_coverage(meta_path="/share/data/camera/deployment/verified_0425.csv"):
    df = pd.read_csv(meta_path)
    dfs = []
    for city, place in list(C.CITIES.items())[:10]:
        print(f"Load building footprint [{place}]..")
        buildings = get_buildings(city, place)
        pano = df.query(f"city == '{city}'")
        print(f"Start coverage calculation ..")
        with multiprocessing.Pool(50) as p:
            rows = list(tqdm(p.imap(get_coverage_df, pano.iterrows()),
                   total=len(pano),
                   smoothing=0.1))
        pano = pd.DataFrame(rows)
        dfs.append(pano)
        pd.concat(dfs).to_csv("/share/data/camera/deployment/verified_0425_coverage.csv", index=False)
