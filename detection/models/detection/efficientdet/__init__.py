import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.structures import Instances, Boxes

from .backbone import EfficientDetWithLoss


class EfficientDetModel(nn.Module):
    """Detectron2 model:
    https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
    """

    def __init__(self, compound_coef, model_args=None):
        super().__init__()
        num_classes = model_args.get("num_classes", None)
        pretrained = model_args.get("pretrained", False)
        self.max_bbox = model_args.get("max_bbox", 30)

        self.model = EfficientDetWithLoss(num_classes=num_classes,
                                          compound_coef=compound_coef,
                                          load_weights=pretrained)

    @staticmethod
    def to_numpy(v):
        if isinstance(v, np.ndarray):
            return v
        else:
            return v.detach().cpu().numpy()

    def forward(self, x):
        N = len(x)
        imgs = torch.stack([sample['image'].float() for sample in x])
        annotations = np.ones((N, self.max_bbox, 5)) * -1
        for i, sample in enumerate(x):
            instances = sample['instances']
            boxes = self.to_numpy(instances.gt_boxes.tensor)
            class_id = self.to_numpy(instances.gt_classes)
            annotation = np.concatenate([boxes, class_id[:, np.newaxis]], 1)
            if len(class_id) > self.max_bbox:
                annotation = annotation[:self.max_bbox, :]
            annotations[i, :len(class_id), :] = annotation
        annotations = torch.from_numpy(annotations)
        return self.model(imgs, annotations, is_train)

    def infer(self, x):
        imgs = torch.stack([sample['image'].float() for sample in x])
        rois = self.model.infer(imgs)
        outs = []
        for sample_input, sample_output in zip(x, rois):
            instances = Instances(
                (sample_input['height'], sample_input['width']))
            instances.pred_boxes = Boxes(sample_output['rois'])
            instances.scores = torch.tensor(sample_output['scores'])
            instances.pred_classes = torch.tensor(sample_output['class_ids'])
            outs.append({"instances": instances})
        return outs


class EfficientDetD0(EfficientDetModel):
    def __init__(self, model_args=None):
        super().__init__(0, model_args)


class EfficientDetD1(EfficientDetModel):
    def __init__(self, model_args=None):
        super().__init__(1, model_args)


class EfficientDetD2(EfficientDetModel):
    def __init__(self, model_args=None):
        super().__init__(2, model_args)


class EfficientDetD3(EfficientDetModel):
    def __init__(self, model_args=None):
        super().__init__(3, model_args)


class EfficientDetD4(EfficientDetModel):
    def __init__(self, model_args=None):
        super().__init__(4, model_args)


class EfficientDetD5(EfficientDetModel):
    def __init__(self, model_args=None):
        super().__init__(5, model_args)


class EfficientDetD6(EfficientDetModel):
    def __init__(self, model_args=None):
        super().__init__(6, model_args)


class EfficientDetD7(EfficientDetModel):
    def __init__(self, model_args=None):
        super().__init__(7, model_args)


class EfficientDetD7X(EfficientDetModel):
    def __init__(self, model_args=None):
        super().__init__(8, model_args)
