import json
import os
from os.path import join
from PIL import Image


LIGHTNING_CKPT_PATH = 'lightning_logs/version_0/checkpoints/'
LIGHTNING_TB_PATH = 'lightning_logs/version_0/'
LIGHTNING_METRICS_PATH = 'lightning_logs/version_0/metrics.csv'


class Args(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__.update(args[0])

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            AttributeError("No such attribute: " + name)


def init_exp_folder(args):
    save_dir = os.path.abspath(args.get("save_dir"))
    exp_name = args.get("exp_name")
    exp_path = join(save_dir, exp_name)
    exp_metrics_path = join(exp_path, "metrics.csv")
    exp_tb_path = join(exp_path, "tb")
    global_tb_path = args.get("tb_path")
    global_tb_exp_path = join(global_tb_path, exp_name)
    if os.environ.get('LOCAL_RANK') is not None:
        return

    # init exp path
    if os.path.exists(exp_path):
        raise FileExistsError(f"Experiment path [{exp_path}] already exists!")
    os.makedirs(exp_path, exist_ok=True)

    os.makedirs(global_tb_path, exist_ok=True)
    if os.path.exists(global_tb_exp_path):
        raise FileExistsError(f"Experiment exists in the global "
                              f"Tensorboard path [{global_tb_path}]!")
    os.makedirs(global_tb_path, exist_ok=True)

    # dump hyper-parameters/arguments
    with open(join(save_dir, exp_name, "args.json"), "w") as f:
        json.dump(args, f)

    # ln -s for metrics
    os.symlink(join(exp_path, LIGHTNING_METRICS_PATH),
               exp_metrics_path)

    # ln -s for tb
    os.symlink(join(exp_path, LIGHTNING_TB_PATH), exp_tb_path)
    os.symlink(exp_tb_path, global_tb_exp_path)

def get_concat_h_cut(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, min(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
