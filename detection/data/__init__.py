import pandas as pd
import os

from .version import Version
from .base import BaseDataset
from .info import DatasetInfo
from . import constants as C


def get_dataset(split="train"):
    meta = pd.read_csv("../data/meta.csv")
    info = DatasetInfo.load("../data/info.yaml")
    return BaseDataset(info, meta)[split]
