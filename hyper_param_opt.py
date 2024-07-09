import json
import logging
import os
import sys

import numpy as np
from hyperopt import fmin, hp, tpe, STATUS_OK, STATUS_FAIL
from hyperopt.exceptions import AllTrialsFailed
from hyperopt.mongoexp import MongoTrials

import train_ce as tce
from main import setup_logging

logger = logging.getLogger('main')
setup_logging("hyper_param_opt.log")
SERVER = "pcknot6.fit.vutbr.cz"
DB_ADDRESS = f"mongo://{SERVER}:1234/ce/jobs"
DB_KEY = "trecdl22-crossencoder-debertav3"


def obj(hpt_config):
    PATH = "/mnt/data/xjarol06_firllm/multidoc2dial-reranking"
    os.chdir(PATH)
    sys.path.append(PATH)

    save_model_name = f"CE_lr{hpt_config['lr']:.2f}_bs{hpt_config['batch_size']:.2f}.pt"
    setup_logging(save_model_name)

    fixed_config = {
        "num_epochs": 10,
        "stop_time": None,
        "load_model_path": None,
        "save_model_path": save_model_name,
        "bert_model_name": "naver/trecdl22-crossencoder-debertav3",
        "train_data_path": "data/DPR_pairs/DPR_pairs_train_50-60.json",
        "test_data_path": "data/DPR_pairs/DPR_pairs_test.jsonl",
        "test_every": "epoch",
    }
    config = {**fixed_config, **hpt_config}
    logger.info("Running with config:")
    logger.info(json.dumps(config, indent=4, sort_keys=True))

    min_mrr = tce.train_ce(**config)
    if min_mrr is None:
        result = {
            "status": STATUS_FAIL,
            "loss": 0
        }
    else:
        result = {
            "status": STATUS_OK,
            "loss": -min_mrr.best_mrr,
            "mrr": min_mrr.best_mrr,
            "at_epoch": min_mrr.best_mrr_epoch,
            "at_gs": min_mrr.best_mrr_gs,
        }

    logger.info(f"Result: {result}")
    trials = MongoTrials(DB_ADDRESS, exp_key=DB_KEY)
    try:
        logger.info("Current best:")
        logger.info(trials.argmin)
    except AllTrialsFailed:
        logger.info("No successful trials yet")
    return result


def run_hyperparam_opt():
    trials = MongoTrials(DB_ADDRESS, exp_key=DB_KEY)
    try:
        space = {
            "label_smoothing": hp.uniform("label_smoothing", low=0.0, high=0.25),
            "dropout_rate": hp.uniform("dropout_rate", low=0.0, high=0.4),
            "weight_decay": hp.uniform("weight_decay", low=0.0, high=0.02),
            "positive_weight": hp.uniform("positive_weight", low=1, high=8),
            "lr": hp.loguniform("lr", low=np.log(1e-6), high=np.log(2e-4)),
            "lr_min": hp.loguniform("lr_min", low=np.log(1e-7), high=np.log(1e-6)),
            "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
            "warmup_percent": hp.uniform("warmup_percent", low=0.05, high=0.2),
            "nr_restarts": hp.choice("nr_restarts", [1, 2, 3, 4, 5]),
            "gradient_clip": hp.choice("gradient_clip", [0, 1, 2]),
        }
        best = fmin(obj, space, trials=trials, algo=tpe.suggest, max_evals=30)
        logger.info("#" * 20)
        logger.info(best)
    except KeyboardInterrupt as e:
        logger.info("INTERRUPTED")
        logger.info("#" * 20)
        logger.info(trials.argmin)


if __name__ == "__main__":
    run_hyperparam_opt()
