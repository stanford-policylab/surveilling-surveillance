import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.structures import Instances, Boxes

from .backbone import Darknet
from .utils import (xy_to_cxcy,
                    non_max_suppression)
from . import constants as C


class YOLOv3Model(nn.Module):
    """YOLO V3 model:
    https://github.com/eriklindernoren/PyTorch-YOLOv3.git
    """

    def __init__(self, cfg_name, model_args=None):
        super().__init__()
        num_classes = model_args.get("num_classes", None)
        self.conf_threshold = model_args.get("conf_threshold", 0.8)
        self.nms_threshold = model_args.get("nms_threshold", 0.4)
        pretrained = model_args.get("pretrained", False)
        ignore_width = model_args.get("ignore_width", 0)
        cfg_path = C.CONFIGS[cfg_name]
        self.model = Darknet(cfg_path,
                             num_classes=num_classes,
                             ignore_width=ignore_width)

    @staticmethod
    def to_numpy(v):
        if isinstance(v, np.ndarray):
            return v
        else:
            return v.detach().cpu().numpy()

    def forward(self, x):
        """
        To N x (img_id, class_id, cx, cy, w, h) format
        """
        N = len(x)
        imgs = torch.stack([sample['image'].float() for sample in x])
        width = imgs.shape[2]
        height = imgs.shape[3]
        if height != 416 or width != 416:
            raise ValueError(
                f"Input images must of size 416 x 416 but is {width} x {height}")

        annotations = []
        for i, sample in enumerate(x):
            instances = sample['instances']
            boxes = self.to_numpy(instances.gt_boxes.tensor)
            class_ids = self.to_numpy(instances.gt_classes)
            for class_id, box in zip(class_ids, boxes):
                cx, cy, w, h = xy_to_cxcy(box, width, height)
                annotations.append([i, class_id, cx, cy, w, h])
        annotations = np.stack(annotations, 0)
        annotations = torch.from_numpy(annotations).float()
        return self.model(imgs, annotations)[0]

    def infer(self, x):
        """
        From N x (xmin, ymin, xmax, ymax, conf, cls_conf_1, cls_conf_2, ..., cls_conf_k) format
        """
        imgs = torch.stack([sample['image'].float() for sample in x])
        width = imgs.shape[2]
        height = imgs.shape[3]
        if height != 416 or width != 416:
            raise ValueError(
                f"Input images must of size 416 x 416 but is {width} x {height}")
        rois = self.model.infer(imgs)
        rois = non_max_suppression(rois,
                                   self.conf_threshold,
                                   self.nms_threshold)
        outs = []
        for sample_input, sample_output in zip(x, rois):
            instances = Instances(
                (sample_input['height'], sample_input['width']))
            print(sample_output)
            if sample_output is not None and len(sample_output):
                instances.pred_boxes = Boxes(sample_output[:, :4])
                instances.scores = torch.tensor(sample_output[:, 4])
                class_conf, class_id = sample_output[:, 5:].max(1)
                instances.pred_classes = torch.tensor(class_id)
            outs.append({"instances": instances})
        return outs


class YOLOv3(YOLOv3Model):
    def __init__(self, model_args=None):
        super().__init__("yolov3", model_args)


class YOLOv3Tiny(YOLOv3Model):
    def __init__(self, model_args=None):
        super().__init__("yolov3-tiny", model_args)
