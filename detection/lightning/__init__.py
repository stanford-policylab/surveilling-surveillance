import torch

from .detection import DetectionTask
from .util import get_ckpt_callback, get_early_stop_callback
from .util import get_logger


def get_task(args):
    return DetectionTask(args)

def load_task(ckpt_path, **kwargs):
    args = torch.load(ckpt_path, map_location='cpu')['hyper_parameters']
    return DetectionTask.load_from_checkpoint(ckpt_path, **kwargs)
