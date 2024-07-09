
import json
import logging
import math
import os
import pickle
import socket
import sys
import traceback

import numpy as np
import torch
import torch.multiprocessing as torch_mp
from hyperopt import STATUS_FAIL, fmin, hp, tpe
from hyperopt.exceptions import AllTrialsFailed
from hyperopt.mongoexp import MongoTrials

from subtask2.scripts.run_t5 import ddp_ce_extractor_fit
from subtask2.trainer.t5trainer import T5Trainer
from subtask2.utils.utility import setup_logging

logger = logging.getLogger(__name__)
SERVER = "pcknot6.fit.vutbr.cz"
DB_ADDRESS = f"mongo://{SERVER}:1234/ce_t5_db/jobs"
DB_KEY = "ce_t5_base"


def obj(hpt_config):
    PATH = "YOUR-PATH-TO-REPOSITORY-ROOT"
    os.chdir(PATH)
    sys.path.append(PATH)



    trials = MongoTrials(DB_ADDRESS, exp_key=DB_KEY)
    try:
        logger.info("Current best:")
        logger.info(trials.argmin)
    except AllTrialsFailed:
        logger.info("No successfull trials yet")
    return result


def run_hyperparam_opt():
    trials = MongoTrials(DB_ADDRESS, exp_key=DB_KEY)
    try:
        space = {
            "hidden_dropout": hp.uniform("hidden_dropout", low=0.0, high=0.5),
            "attention_dropout": hp.uniform("attention_dropout", low=0.0, high=0.5),
            "learning_rate": hp.loguniform(
                "learning_rate", low=np.log(1e-6), high=np.log(2e-4)
            ),
            "weight_decay": hp.uniform("weight_decay", low=0.0, high=3e-2),
            "true_batch_size": hp.quniform("true_batch_size", low=16, high=80, q=16),
            "scheduler_warmup_proportion": hp.uniform(
                "scheduler_warmup_proportion", low=0.0, high=0.2
            ),
            "scheduler": hp.choice("scheduler", ["linear", "constant", None]),
        }
        best = fmin(obj, space, trials=trials, algo=tpe.suggest, max_evals=1000)
        logger.info("#" * 20)
        logger.info(best)
    except KeyboardInterrupt as e:
        logger.info("INTERRUPTED")
        logger.info("#" * 20)
        logger.info(trials.argmin)

if __name__ == "__main__":
    run_hyperparam_opt()
