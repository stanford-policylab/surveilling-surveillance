import nni
import pickle as pkl 
import json
import pytorch_lightning as pl
import os
import numpy as np
import torch
import torchvision
from PIL import Image
import pandas as pd
from detectron2.data import transforms as T
from detectron2.structures import Instances, Boxes
from detectron2.utils.visualizer import Visualizer
from torch.utils.data import DataLoader
from ignite.metrics import Accuracy

from models import get_model
from eval import DetectionEvaluator
from data import get_dataset
from util import constants as C
from util import get_concat_h_cut
from .logger import TFLogger


class DetectionTask(pl.LightningModule, TFLogger):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        self.model = get_model(params)
        self.evaluator = DetectionEvaluator()

    def training_step(self, batch, batch_nb):
        losses = self.model.forward(batch)
        loss = torch.stack(list(losses.values())).mean()
        return loss

    def validation_step(self, batch, batch_nb):
        losses = self.model.forward(batch)
        loss = torch.stack(list(losses.values())).mean()
        preds = self.model.infer(batch)
        self.evaluator.process(batch, preds)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log("val_loss", avg_loss)
        metrics = self.evaluator.evaluate()
        nni.report_intermediate_result(metrics['mAP']) 
        self.evaluator.reset()
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch, batch_nb):
        preds = self.model.infer(batch)
        conf_threshold = self.hparams.get("conf_threshold", 0)
        iou_threshold = self.hparams.get("iou_threshold", 0.5)
        padding = self.hparams.get("padding", 10)
        if self.hparams.get('visualize', False) or self.hparams.get("deploy", False):
            for i, (sample, pred) in enumerate(zip(batch, preds)):
                instances = pred['instances']
                boxes = instances.get('pred_boxes').tensor
                class_id = instances.get('pred_classes')

                # Filter by scores
                scores = instances.scores
                keep_id_conf = scores > conf_threshold
                boxes_conf = boxes[keep_id_conf]
                scores_conf = scores[keep_id_conf]
                class_id_conf = class_id[keep_id_conf]
                if boxes_conf.size(0) == 0:
                    continue
                
                # Filter by nms
                keep_id_nms = torchvision.ops.nms(boxes_conf,
                                                  scores_conf, 
                                                  iou_threshold)

                boxes_nms = boxes_conf[keep_id_nms]
                scores_nms = scores_conf[keep_id_nms]
                class_id_nms = class_id_conf[keep_id_nms]

                # Pad box size
                boxes_nms[:, 0] -= padding
                boxes_nms[:, 1] -= padding
                boxes_nms[:, 2] += padding
                boxes_nms[:, 3] += padding
                boxes_nms = torch.clip(boxes_nms, 0, 640)

                for j in range(len(scores_nms)):
                    instances = Instances((640, 640)) 
                    class_id_numpy = class_id_nms.to("cpu").numpy()[j]
                    box_numpy = boxes_nms.to("cpu").numpy()[j]
                    score_numpy = scores_nms.to("cpu").numpy()[j]

                    instances.pred_classes = np.array([class_id_numpy])
                    instances.pred_boxes = Boxes(box_numpy[np.newaxis,:])
                    instances.scores = np.array([score_numpy])
                    
                    v = Visualizer(np.transpose(sample['image'].to("cpu"), (1,2,0)), 
                                   instance_mode=1, 
                                   metadata=C.META)
                    out = v.draw_instance_predictions(instances)
                    img_box = Image.fromarray(out.get_image())

                    if self.hparams.get("deploy", False): 
                        panoid = sample['panoid']
                        heading = sample['heading']
                        save_path = f".output/{panoid[:2]}/{panoid}_{heading}_{j}.jpg"
                        json_save_path = f".output/{panoid[:2]}/{panoid}_{heading}_{j}.json"
                        
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        img_org = Image.open(sample['save_path']) 
                        img_out = get_concat_h_cut(img_org, img_box) 
                        img_out.save(save_path)
                        data = {"panoid": panoid, 
                               "heaidng": int(heading), 
                               "detection_id": int(j),
                               "class_id": int(class_id_numpy),
                               "box": [int(x) for x in box_numpy],
                               "score": float(score_numpy),
                               "save_path": save_path}
                        with open(json_save_path, 'w') as fp:
                                json.dump(data, fp)
                    else:
                        img_box.save(f"outputs/{batch_nb}_{i}.jpg")
                    
        self.evaluator.process(batch, preds)

    def test_epoch_end(self, outputs):
        metrics = self.evaluator.evaluate()
        nni.report_final_result(metrics['mAP'])
        self.log_dict(metrics)

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])]

    def train_dataloader(self):
        dataset = get_dataset('train')
        return dataset.detection_dataloader(
                          shuffle=True,
                          augmentations=[
                                  T.RandomBrightness(0.9, 1.1),
                                  T.RandomFlip(prob=0.5),
                              ], 
                          batch_size=self.hparams['batch_size'], 
                          num_workers=8)

    def val_dataloader(self):
        dataset = get_dataset('valid')
        return dataset.detection_dataloader(
                          shuffle=False,
                          batch_size=1,
                          num_workers=8)

    def test_dataloader(self):
        if self.hparams.get('deploy', False):
            dataset = load_dataset(self.hparams['dataset_name'])
            df = pd.read_csv(self.hparams['deploy_meta_path']).query("downloaded == True")
            df["image_id"] = df['save_path']
            df["gsv_image_path"] = df['save_path']
            df['annotations'] = "[]"
            dataset._meta = df
            return dataset.detection_dataloader(
                          shuffle=False,
                          batch_size=self.hparams.get("test_batch_size", 1),
                          num_workers=8)
        else:
            test_split = self.hparams.get("test_split", "valid") 
            dataset = get_dataset(test_split)
            return dataset.detection_dataloader(
                          shuffle=False,
                          batch_size=1,
                          num_workers=8)
