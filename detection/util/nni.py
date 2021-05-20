import os
import nni
import time
import logging
import json
import traceback
from glob import glob


def _cast_value(v):
    if v == "True":
        v = True
    elif v == "False":
        v = False
    elif v == "None":
        v = None
    return v


def run_nni(train_func, test_func):
    try:
        params = nni.get_next_parameter()
        params = {k: _cast_value(v) for k, v in params.items()}
        params['exp_name'] = "nni" + str(time.time())
        logging.info("Final Params:")
        logging.info(params)

        save_dir, exp_name = train_func(**params)
        ckpt_reg = os.path.join(save_dir, exp_name, "*.ckpt")
        print(ckpt_reg)
        ckpt_path = list(glob(ckpt_reg))[-1]
        
        test_func(ckpt_path=ckpt_path)

    except RuntimeError as re:
        if 'out of memory' in str(re):
            time.sleep(600)
            params['batch_size'] = int(0.5 * params['batch_size'])
            train(**params)
        else:
            traceback.print_exc()
            nni.report_final_result(-1)
    except Exception as e:
        traceback.print_exc()
        nni.report_final_result(-2)
