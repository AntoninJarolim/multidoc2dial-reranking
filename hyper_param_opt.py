import logging
import os
import socket
import sys

import numpy as np
from hyperopt import fmin, hp, tpe, STATUS_OK, STATUS_FAIL
from hyperopt.exceptions import AllTrialsFailed
from hyperopt.mongoexp import MongoTrials

import train_ce as tce
from main import setup_logging

logger = logging.getLogger('main')
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
        "save_model": True,
        "evaluate_before_training": socket.gethostname() == SERVER.split(".")[0],
        "evaluation_take_n": 50,
    }

    data_args = tce.TrainDataArgs(load_model_path=fixed_config["load_model_path"],
                                  save_model_path=fixed_config["save_model_path"],
                                  bert_model_name=fixed_config["bert_model_name"],
                                  train_data_path=fixed_config["train_data_path"],
                                  test_data_path=fixed_config["test_data_path"],
                                  dont_save_model=fixed_config["save_model"],
                                  )

    train_args = tce.TrainHyperparameters(num_epochs=fixed_config["num_epochs"],
                                          lr=hpt_config["lr"],
                                          weight_decay=hpt_config["weight_decay"],
                                          positive_weight=hpt_config["positive_weight"],
                                          dropout_rate=hpt_config["dropout_rate"],
                                          label_smoothing=hpt_config["label_smoothing"],
                                          gradient_clip=hpt_config["gradient_clip"],
                                          batch_size=hpt_config["batch_size"],
                                          evaluation_take_n=fixed_config["evaluation_take_n"],
                                          lr_min=hpt_config["lr_min"],
                                          warmup_percent=hpt_config["warmup_percent"],
                                          nr_restarts=hpt_config["nr_restarts"],
                                          )
    min_mrr = tce.train_ce(data_args, train_args)
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
        # np arrays of ints to prevent hyperopt auto conversion to float
        # ints of type object to make it json serializable
        space = {
            "label_smoothing": hp.uniform("label_smoothing", low=0.0, high=0.25),
            "dropout_rate": hp.uniform("dropout_rate", low=0.0, high=0.4),
            "weight_decay": hp.uniform("weight_decay", low=0.0, high=0.02),
            "positive_weight": hp.uniform("positive_weight", low=1, high=8),
            "lr": hp.loguniform("lr", low=np.log(1e-6), high=np.log(2e-4)),
            "lr_min": hp.loguniform("lr_min", low=np.log(1e-7), high=np.log(1e-6)),
            "batch_size": hp.choice("batch_size", np.array([16, 32, 64, 128], dtype=object)),
            "warmup_percent": hp.uniform("warmup_percent", low=0.05, high=0.2),
            "nr_restarts": hp.choice("nr_restarts", np.array([1, 2, 3, 4, 5], dtype=object)),
            "gradient_clip": hp.choice("gradient_clip", np.array([0, 1, 2], dtype=object)),
        }
        best = fmin(obj, space, trials=trials, algo=tpe.suggest, max_evals=30)
        logger.info("#" * 20)
        logger.info(best)
    except KeyboardInterrupt:
        logger.info("INTERRUPTED")
        logger.info("#" * 20)
        logger.info(trials.argmin)


if __name__ == "__main__":
    run_hyperparam_opt()
