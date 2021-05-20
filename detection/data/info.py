import yaml
import dataclasses
import pandas as pd
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union

from .version import Version


class BaseInfo:
    @classmethod
    def from_dict(cls, dataset_info_dict: dict) -> "DatasetInfo":
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: v for k, v in dataset_info_dict.items() if k in field_names})


@dataclass
class ImageSourceInfo(BaseInfo):
    # Required Fields
    name: str = field(default_factory=str)
    height: int = field(default_factory=int)
    width: int = field(default_factory=int)
    date: str = field(default_factory=str)
    # Optional Fields
    channels: Optional[list] = None
    resolution: Optional[str] = field(default_factory=str)


@dataclass
class DatasetInfo(BaseInfo):
    name: str = field(default_factory=str)
    description: str = field(default_factory=str)
    author: str = field(default_factory=str)
    version: Union[str, Version] = field(default_factory=Version)
    date: str = field(default_factory=str)
    task: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    sources: List[ImageSourceInfo] = field(default_factory=ImageSourceInfo)

    def __post_init__(self):
        if self.version is not None and not isinstance(self.version, Version):
            if isinstance(self.version, str):
                self.version = Version(self.version)
            else:
                self.version = Version.from_dict(self.version)
        if self.sources is not None and not all(
                [isinstance(s, ImageSourceInfo) for s in self.sources]):
            sources = []
            for source in self.sources:
                if isinstance(source, ImageSourceInfo):
                    pass
                elif isinstance(source, dict):
                    source = ImageSourceInfo.from_dict(source)
                else:
                    raise ValueError(
                        f"Unknown type for ImageSourceInfo: {type(source)}")
                sources.append(source)
            self.sources = sources

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            yaml_dict = yaml.load(f, Loader=yaml.SafeLoader)
        return cls.from_dict(yaml_dict)

    def save(self, path):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f)

    def dump(self, fileobj):
        yaml.dump(asdict(self), fileobj)


class DatasetInfoMixin:

    def __init__(self,
                 info: DatasetInfo,
                 meta: pd.DataFrame,
                 split: Optional[str] = None):
        self._info = info
        self._meta = meta
        self._split = split
        self._format = None

        if self._split is not None and self._split != 'all':
            self._meta.query(f"split == '{self._split}'", inplace=True)

    def __len__(self):
        return len(self._meta)

    def __repr__(self):
        features = self.features
        if len(features) < 5:
            features_repr = "[" + ", ".join(features) + "]"
        else:
            features_repr = "[" + \
                ", ".join(features[:3] + ["...", features[-1]]) + "]"
        return f"{type(self).__name__}(split: {self.split}, version: {self.version}, features[{len(features)}]: {features_repr}, samples: {self.__len__()})"

    def get_split(self, split):
        if split == "all":
            return self
        elif split in self.splits:
            result = self.query(f"split == '{split}'")
            result._split = split
            return result
        else:
            raise ValueError(
                f"Unknown split {split}. Split has to be one of {list(self.splits.keys())}")

    def slice(self, expr):
        result = deepcopy(self)
        result._meta = result._meta.iloc[expr]
        return result

    def query(self, expr):
        result = deepcopy(self)
        result._meta = result._meta.query(expr)
        return result

    def filter(self, func):
        result = deepcopy(self)
        result._meta = result._meta[result._meta.apply(func, 1)].reset_index()
        return result

    def set_format(self, columns: Union[dict, list]):
        self._format = columns

    def reset_format(self):
        self.set_format(None)

    def value_counts(self, value):
        return self._meta[value].value_counts().to_dict()

    @property
    def info(self):
        return self._info

    @property
    def meta(self):
        return self._meta.copy()

    @property
    def name(self):
        return self._info.name

    @property
    def version(self):
        return self._info.version

    @property
    def description(self):
        return self._info.description

    @property
    def author(self):
        return self._info.author

    @property
    def sources(self):
        return [s.name for s in self._info.sources]

    @property
    def split(self):
        if self._split is None:
            return "all"
        return self._split

    @property
    def splits(self):
        return self.value_counts("split")

    @property
    def features(self):
        features = list(self._meta.columns)
        return features
