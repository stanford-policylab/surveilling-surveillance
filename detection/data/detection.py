import torch
from torch.utils.data import Dataset

from .info import DatasetInfoMixin
from . import constants as C


def trivial_batch_collator(batch):
    return batch


class DetectionMixin:
    def detection_dataloader(self,
                             augmentations=None,
                             is_train=True,
                             use_instance_mask=False,
                             image_path_col=None,
                             **kwargs):
        from detectron2.data import DatasetMapper
        if augmentations is None:
            augmentations = []
        mapper = DatasetMapper(is_train=is_train,
                               image_format="RGB",
                               use_instance_mask=use_instance_mask,
                               instance_mask_format="bitmask",
                               augmentations=augmentations
                               )
        return DetectionDataset(info=self.info,
                                meta=self.meta,
                                split=self.split,
                                image_path_col=image_path_col,
                                mapper=mapper) \
            .dataloader(**kwargs)


class DetectionDataset(Dataset, DatasetInfoMixin):
    """
    Dataset class that provides standard Detectron2 model input format:
    https://detectron2.readthedocs.io/en/latest/tutorials/models.html?highlight=input%20format#model-input-format
    Notice the annotation column in the meta file need to follow Detectron2's
    standard dataset dict format:
    https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#standard-dataset-dicts
    """

    def __init__(self, info, meta, mapper, split=None, image_path_col=None):
        if C.ANNOTATION_COLUMN not in meta.columns:
            raise ValueError(f"[{C.ANNOTATION_COLUMN}] column not found in the meta data.")

        if image_path_col is None:
            image_path_cols = [
                c for c in meta.columns if c.endswith("image_path")]
            if len(image_path_cols) == 0:
                raise ValueError(
                    "No image path column found in the meta data. Please check meta data and use `image_path_col` argument to specify the column.")
            elif len(image_path_cols) > 1:
                raise ValueError(
                    "Multiple image path columns found in the meta data. Please use `image_path_col` argument to specify the column.")
            else:
                image_path_col = image_path_cols[0]

        meta = meta.rename(columns={image_path_col: "file_name"})

        self.mapper = mapper

        DatasetInfoMixin.__init__(self,
                                  info=info,
                                  meta=meta,
                                  split=split)

    def __getitem__(self, index):
        sample = self._meta.iloc[index].to_dict()
        sample[C.ANNOTATION_COLUMN] = eval(sample[C.ANNOTATION_COLUMN])
        return self.mapper(sample)

    def dataloader(self, **kwargs):
        return torch.utils.data.DataLoader(
            self,
            collate_fn=trivial_batch_collator,
            **kwargs)
