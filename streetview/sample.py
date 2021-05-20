import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import os
from geopy.distance import distance
from shapely.geometry import MultiPoint 

from .util import get_heading

def random_points(edges, 
                  n=100, 
                  d=None, 
                  verbose=False):
    m = len(edges)
    lengths = edges['length'].tolist()
    total_length = edges.sum()['length']
    lengths_normalized = [l/total_length for l in lengths] 
    
    rows = []
    points = []
    indices = np.random.choice(range(m), 
                               size=2*n,
                               p=lengths_normalized)
    pbar = tqdm(total=n)
    i = j = 0
    while i < n:
        index = indices[j]
        row = edges.iloc[index]
        u, v, key = edges.index[index]
        line = row['geometry']
        offset = np.random.rand() * line.length
        point = line.interpolate(offset)
        lat = point.y
        lon = point.x
        flag = 1
        if d is not None:
            for _lat, _lon in points:
                _d = np.sqrt(np.square(lat-_lat) + np.square(lon-_lon))
                if _d < 1e-4 and distance((lat, lon), (_lat, _lon)).m < d:
                    flag = 0
                    break
        if flag:
            i += 1
            pbar.update(1)
            start = line.interpolate(offset*0.9)
            end = line.interpolate(min(line.length, offset*1.1))
            heading = get_heading(start.y, start.x, end.y, end.x)
            rows.append({"lat": lat,
                         "lon": lon,
                         "id": i,
                         "u": u,
                         "v": v,
                         "heading": heading,
                         "offset": offset,
                         "key": key})
            points.append((lat, lon))
        j += 1
    pbar.close()
    return pd.DataFrame(rows)

def random_stratified_points(edges, n=10):
    m = len(edges)
    rows = []
    for index in range(len(edges)):
        row = edges.iloc[index]
        u, v, key = edges.index[index]
        line = row['geometry']
        
        for _ in range(n):
            offset = np.random.rand() * line.length
            point = line.interpolate(offset)
            lat = point.y
            lon = point.x
            rows.append({"lat": lat,
                         "lon": lon,
                         "u": u,
                         "v": v,
                         "key": key})
    return pd.DataFrame(rows)
    
def select_panoid(meta, 
                    n=5000, 
                    distance=10, 
                    selection="closest",
                    seed=123):
    YEARS = ["2010<year<2016", "2016<=year"]
    
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)

    # Filter by distance
    meta = meta.query(f"distance < {distance}")
    
    # Filter by occurance for both pre and post
    meta_pre = meta.query(YEARS[0]).drop_duplicates(["lat_anchor", "lon_anchor"])
    meta_post = meta.query(YEARS[1]).drop_duplicates(["lat_anchor", "lon_anchor"])    
    meta_both = meta_pre.merge(meta_post, on=["lat_anchor", "lon_anchor"], how="inner")
    
    # Sample anchor points 
    meta_sample = meta_both.drop_duplicates(['lat_anchor', 'lon_anchor']).sample(n, replace=False)
    lat_anchor_chosen = meta_sample.lat_anchor.unique()
    lon_anchor_chosen = meta_sample.lon_anchor.unique()

    # Sample for pre and post
    meta_sub = meta[meta.lat_anchor.isin(lat_anchor_chosen)]
    meta_sub = meta_sub[meta_sub.lon_anchor.isin(lon_anchor_chosen)]

    # Select panoid
    groups = []
    for years in YEARS:
        group = meta_sub.query(years)
        if selection == "closest":
            group = group.sort_values(['lat_anchor','lon_anchor', 'distance']) 
        else:
            group = group.sort_values(['lat_anchor','lon_anchor', 'year'], ascending=False) 
        group = group.groupby(['lat_anchor','lon_anchor']).first().reset_index()        
        group['year'] = group.year.apply(int)
        groups.append(group)
    
    # Random select the orthogonal heading
    merged = groups[0].merge(groups[1], 
                             on=['lat_anchor', 'lon_anchor', 'u', 'v', 'key', 'heading', 'offset'], 
                             suffixes=("_pre", "_post"))
    
    merged['heading_pre'] = merged['heading_post'] = (merged.heading + 360 + 90 - 180 * (np.random.rand(n) > 0.5)) % 360
    merged['heading_pre'] = merged['heading_pre'].apply(int)
    merged['heading_post'] = merged['heading_post'].apply(int)
    return merged

def select_panoid_recent(meta,
                    year,
                    n=5000, 
                    distance=10,
                    seed=123):
    
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)

    # Filter by distance
    meta = meta.query(f"distance < {distance}")
    meta = meta.query(f"year >= {year}")

    # Sample anchor points 
    meta_sample = meta.drop_duplicates(['id']).sample(n, replace=False)
    lat_anchor_chosen = meta_sample.lat_anchor.unique()
    lon_anchor_chosen = meta_sample.lon_anchor.unique()

    # Sample for pre and post
    meta_sub = meta[meta.lat_anchor.isin(lat_anchor_chosen)]
    meta_sub = meta_sub[meta_sub.lon_anchor.isin(lon_anchor_chosen)]

    # Select panoid

    meta = meta_sub.sort_values(['lat_anchor','lon_anchor', 'distance']) \
                         .groupby(['lat_anchor','lon_anchor']) \
                         .first().reset_index()     

    # Random select the orthogonal heading
    meta['road_heading'] = meta.heading
    meta['heading'] = (meta.heading + 360 + 90 - 180 * (np.random.rand(n) > 0.5)) % 360
    meta['heading'] = meta['heading'].apply(int)
    meta['year'] = meta['year'].apply(int)
    meta['month'] = meta['month'].apply(int)
    meta['save_path'] = meta.apply(get_path, 1)
    return meta

def get_path(row):
    panoid = row['panoid']
    heading = row['heading']
    return os.path.join("/scratch/haosheng/camera/", panoid[:2], panoid[2:4], panoid[4:6], panoid[6:], f"{heading}.png")
