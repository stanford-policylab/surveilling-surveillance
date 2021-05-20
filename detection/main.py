import os
import fire
from pytorch_lightning import Trainer

from util.nni import run_nni
from util import init_exp_folder, Args
from util import constants as C
from lightning import (get_task,
                       load_task,
                       get_ckpt_callback, 
                       get_early_stop_callback,
                       get_logger)


def train(save_dir=C.SANDBOX_PATH,
          tb_path=C.TB_PATH,
          exp_name="DemoExperiment",
          model="FasterRCNN",
          task='detection',
          gpus=1,
          pretrained=True,
          batch_size=8,
          accelerator="ddp",
          gradient_clip_val=0.5,
          max_epochs=100,
          learning_rate=1e-5,
          patience=30,
          limit_train_batches=1.0,
          limit_val_batches=1.0,
          limit_test_batches=1.0,
          weights_summary=None,
          ):
    """
    Run the training experiment.

    Args:
        save_dir: Path to save the checkpoints and logs
        exp_name: Name of the experiment
        model: Model name
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
        pretrained: Whether or not to use the pretrained model
        num_classes: Number of classes
        accelerator: Distributed computing mode
        gradient_clip_val:  Clip value of gradient norm
        limit_train_batches: Proportion of training data to use
        max_epochs: Max number of epochs
                patience: number of epochs with no improvement after
                                  which training will be stopped.
        tb_path: Path to global tb folder
        loss_fn: Loss function to use
        weights_summary: Prints a summary of the weights when training begins.

    Returns: None

    """
    num_classes = 2
    dataset_name = "camera-detection-new"

    args = Args(locals())
    init_exp_folder(args)
    task = get_task(args)
    trainer = Trainer(gpus=gpus,
                      accelerator=accelerator,
                      logger=get_logger(save_dir, exp_name),
                      callbacks=[get_early_stop_callback(patience),
                                 get_ckpt_callback(save_dir, exp_name, monitor="mAP", mode="max")],
                      weights_save_path=os.path.join(save_dir, exp_name),
                      gradient_clip_val=gradient_clip_val,
                      limit_train_batches=limit_train_batches,
                      limit_val_batches=limit_val_batches,
                      limit_test_batches=limit_test_batches,
                      weights_summary=weights_summary,
                      max_epochs=max_epochs)
    trainer.fit(task)
    return save_dir, exp_name


def test(ckpt_path,
         visualize=False,
         deploy=False,
         limit_test_batches=1.0,
         gpus=1,
         deploy_meta_path="/home/haosheng/dataset/camera/deployment/16cityp1.csv",
         test_batch_size=1,
         **kwargs):
    """
    Run the testing experiment.

    Args:
        ckpt_path: Path for the experiment to load
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
    Returns: None

    """
    task = load_task(ckpt_path, 
                     visualize=visualize,
                     deploy=deploy, 
                     deploy_meta_path=deploy_meta_path,
                     test_batch_size=test_batch_size,
                     **kwargs)
    trainer = Trainer(gpus=gpus,
                      limit_test_batches=limit_test_batches)
    trainer.test(task)


def nni():
    run_nni(train, test)


if __name__ == "__main__":
    fire.Fire()
