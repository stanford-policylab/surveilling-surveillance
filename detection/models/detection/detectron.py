import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.utils.events import EventStorage
from detectron2.structures import Instances, Boxes
from detectron2.checkpoint import DetectionCheckpointer


class Detectron2Model(nn.Module):
    """Detectron2 model:
    https://github.com/facebookresearch/detectron2
    """
    MODEL_CONFIG = {
        "mask_rcnn": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "faster_rcnn": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        "retinanet": "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
        "rpn": "COCO-Detection/rpn_R_50_FPN_1x.yaml",
        "fast_rcnn": "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml"}

    def __init__(self, model_name, model_args=None):
        super().__init__()
        num_classes = model_args.get("num_classes", None)
        pretrained = model_args.get("pretrained", False)
        nms_threshold = model_args.get("nms_threshold", 0.5)
        if model_args.get("gpus", None) is None:
            device = "cpu"
        else:
            device = "cuda"

        self.cfg = get_cfg()
        config_path = self.MODEL_CONFIG[model_name]
        self.cfg.merge_from_file(model_zoo.get_config_file(config_path))

        # Update number of classes
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
        
        # Segmentation 
        self.cfg.INPUT.MASK_FORMAT='bitmask'
        
        # NMS
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold
        self.cfg.MODEL.RPN.NMS_THRESH_TEST = nms_threshold

        self.cfg.MODEL.DEVICE = device
        model = build_model(self.cfg)

        # Load pretrained model
        if pretrained:
            DetectionCheckpointer(model).load(
                model_zoo.get_checkpoint_url(config_path))

        self.model = model

    def forward(self, x):
        if self.training:
            with EventStorage() as storage:
                out = self.model(x)
        else:
            self.model.train()
            with torch.no_grad(), EventStorage() as storage:
                out = self.model(x)
            self.model.eval()
        return out

    def infer(self, x):
        with torch.no_grad():
            out = self.model(x)
        return out


class FasterRCNN(Detectron2Model):
    def __init__(self, model_args=None):
        super().__init__('faster_rcnn', model_args)


class MaskRCNN(Detectron2Model):
    def __init__(self, model_args=None):
        super().__init__('mask_rcnn', model_args)


class FastRCNN(Detectron2Model):
    def __init__(self, model_args=None):
        super().__init__('fast_rcnn', model_args)


class RetinaNet(Detectron2Model):
    def __init__(self, model_args=None):
        super().__init__('retinanet', model_args)


class RPN(Detectron2Model):
    def __init__(self, model_args=None):
        super().__init__('rpn', model_args)
