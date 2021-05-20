import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from .info import DatasetInfoMixin
from .detection import DetectionMixin 
from .util import _is_path


class BaseDataset(Dataset,
                  DatasetInfoMixin,
                  DetectionMixin):

    def __init__(self,
                 info,
                 meta,
                 split=None,
                 ):
        DatasetInfoMixin.__init__(self,
                                  info=info,
                                  meta=meta,
                                  split=split)

    @staticmethod
    def _load_image_file(file_path):
        if not _is_path(file_path):
            return None
        image_pil = Image.open(file_path).convert('RGB')
        image_np = np.array(image_pil)
        return image_np

    @staticmethod
    def _load_pickle_file(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def _load_numpy_file(file_path):
        data = np.load(file_path)
        return data

    @classmethod
    def _load_single_image(cls, sample_dict):
        new_sample_dict = {}
        for k, v in sample_dict.items():
            if k.endswith("image_path"):
                new_sample_dict[k.replace(
                    "_image_path", "_image")] = cls._load_image_file(v)
            else:
                new_sample_dict[k] = v
        return new_sample_dict

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.get_split(index)
        elif isinstance(index, slice):
            return self.slice(index)

        sample = self._meta.iloc[index].to_dict()

        # Replace Nan
        # TODO

        # Load Images
        sample = self._load_single_image(sample)

       # Apply Format
        if isinstance(self._format, list):
            sample = {k: v for k, v in sample.items() if k in self._format}
        elif isinstance(self._format, dict):
            sample = {self._format[k]: v for k,
                      v in sample.items() if k in self._format}

        return sample
