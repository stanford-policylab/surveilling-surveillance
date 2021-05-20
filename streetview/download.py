import time
import traceback
import sys
import random
import hashlib
import hmac
import base64
import fire
import os
import logging
import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import requests as r
import urllib.parse as urlparse

from util import constants as C


def _init_downloader(*args):
    global downloader
    downloader = SVImageDownloader(*args)


def _download(key):
    global downloader
    return downloader.download(key)


class SVImageDownloader:
    def __init__(self,
                 key_to_sec,
                 save_dir,
                 sleep_time=0.0):
        self.key_to_sec = key_to_sec
        self.sleep_time = sleep_time
        self.save_dir = save_dir

    def get_url(self, panoid, head, keysec):
        key, secret = keysec
        url = (f"https://maps.googleapis.com/maps/api/streetview?"
               f"size={C.SV_SIZE}&pano={panoid}&fov={C.SV_FOV}&"
               f"heading={head}&pitch={C.SV_PITCH}&key={key}")
        url = urlparse.urlparse(url)

        # We only need to sign the path+query part of the string
        url_to_sign = url.path + "?" + url.query
        # Decode the private key into its binary format
        # We need to decode the URL-encoded private key
        decoded_key = base64.urlsafe_b64decode(secret)

        # Create a signature using the private key and the URL-encoded
        # string using HMAC SHA1. This signature will be binary.
        signature = hmac.new(decoded_key,
                             str.encode(url_to_sign),
                             hashlib.sha1)

        # Encode the binary signature into base64 for use within a URL
        encoded_signature = base64.urlsafe_b64encode(signature.digest())
        original_url = f'{url.scheme}://{url.netloc}{url.path}?{url.query}'

        return original_url + "&signature=" + encoded_signature.decode()

    def download_image(self,
                       panoid,
                       head,
                       keysec,
                       save_path,
                       ):
        os.makedirs(save_path, exist_ok=True)
        url = self.get_url(panoid, head, keysec)
        resp = r.get(url)
        img_binary = resp._content
        write_path = os.path.join(save_path, f'{panoid}_{head}.jpg')
        with open(write_path, "wb+") as f:
            f.write(img_binary)

    def download(self, rtuple):
        rid, row = rtuple
        time.sleep(np.random.rand() * self.sleep_time)
        head = row['heading']
        try:
            key_idx = rid % len(self.key_to_sec)
            keysec = list(self.key_to_sec)[key_idx]
            self.download_image(panoid=row['panoid'], 
                                    head=head, 
                                    keysec=keysec,
                                    save_path=self.save_dir)
        except BaseException as e:
            traceback.print_exception(*sys.exc_info())
            return {"panoid": row['panoid'],
                        "heading": head,
                        "exception": str(e)}
        return {"panoid": None}

class ParallelSVImageDownloader:
    def __init__(self,
                 key_to_sec,
                 save_dir,
                 sleep_time=0.0,
                 nthread=10,
                 ):
        self.key_to_sec = key_to_sec
        self.save_dir = save_dir
        self.sleep_time = sleep_time
        self.nthread = nthread
        os.makedirs(self.save_dir, exist_ok=True)
        
    def download(self, df, sample_frac=1.0):
        df = df.sample(frac=sample_frac)
        
        print("Start downloading ...")
        with mp.Pool(self.nthread,
                 initializer=_init_downloader,
                 initargs=(self.key_to_sec, self.save_dir, self.sleep_time)) as p:
            df = list(tqdm(p.imap(_download, df.iterrows()),
                       total=len(df),
                       smoothing=0.1))

        image_errors = pd.DataFrame(df)
        image_errors.dropna(subset=['panoid'], inplace=True)
        return image_errors


def download_streetview_image(key, sec):
    df = pd.read_csv("data/meta.csv")
    downloader = ParallelSVImageDownloader(key_to_sec=[(key, sec)], 
                                           save_dir="./data/image")
    downloader.download(df)
