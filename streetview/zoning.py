import geopandas as gpd
from geopy import distance
import pandas as pd
from shapely.geometry import Point
import numpy as np
from shapely.ops import nearest_points
from sklearn.neighbors import KDTree
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys

from util import constants as C

CITIES = [('NYC', 'New York'), ('SF', 'San Francisco'), ('Seattle', 'Seattle'), ('Boston', 'Boston'), ('Chicago', 'Chicago'), ('Philadelphia', 'Philadelphia'), ('DC', 'Washington'), 
                  ('LA', 'Los Angeles'), ('Baltimore', 'Baltimore'), ('Milwaukee', 'Milwaukee')]

class Zoning:
    def __init__(self, path):
        self.path = path
        self.gdf = gpd.read_file(self.path)
        self.zone_type = self.gdf.zone_type.tolist()
        self._get_centroids()
        
    def _get_centroids(self):
        centroids = self.gdf.centroid
        coords = []
        for i, c in enumerate(centroids):
            if c is None or self.zone_type[i] == 'roads':
                coords.append([10000, 10000])
            else:
                coords.append([c.y, c.x])
        self.coords = KDTree(np.array(coords), leaf_size=30)
        
    def get_zone(self, lat, lon, n=-1, return_polygon=False):
        if n == -1:
            ind = range(len(self.gdf))
        else:
            ind = self.coords.query(np.array([lat, lon])[np.newaxis,:], k=n, return_distance=False).flatten()
        dist = 10000
        zone_type = None
        zone = None
        for i in list(ind):
            _zone = self.gdf.geometry.iloc[i]
            #for p in nearest_points(_zone, Point(lon, lat)): 
            p = nearest_points(_zone, Point(lon, lat))[0] 
            _lat, _lon = p.y, p.x
            _dist = distance.distance((lat, lon), (_lat, _lon)).m
            if _dist < dist:
                zone_type = self.zone_type[i]
                dist = _dist
                zone = _zone
        if return_polygon:
            return zone_type, dist, zone
        else:
            return zone_type, dist
        
def calculate_zone(meta_path="/share/data/camera/deployment/verified_0425.csv"):

    df = pd.read_csv(meta_path)
    dfs = []
    for city, city_tag in CITIES:
        print(f"Loading zoning shapefile for [{city_tag}]..")
        try:
            zone = Zoning(f"/share/data/camera/zoning/{city_tag}_zoning_clean.shp")
        except Exception as e:
            print(str(e))
            continue

        final = df.query(f"city == '{city}'")
        rows = []
        for rid, row in tqdm(final.iterrows(), total=len(final)):
            z, d = zone.get_zone(row['lat'], row['lon'], n=5)
            row['zone_type'] = z
            row['zone_distance'] = d
            rows.append(row)
        zone_final = pd.DataFrame(rows)
        dfs.append(zone_final)
        pd.concat(dfs).to_csv("/share/data/camera/deployment/verified_0425_zone.csv", index=False)
