import argparse
import logging
import os
from datetime import datetime

import train_ce as tce
from utils import compute_recall

logger = logging.getLogger('main')


def setup_logging(log_suffix=""):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join('logs', f'log_{current_time}_{log_suffix}.log')

    logger.propagate = False  # Prevent the log messages from being duplicated in the python console

    # Remove all handlers associated with the logger object
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    # Log current time to be able to match .pt file with log file
    logger.info(f"Current time: {current_time}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a cross encoder model.")

    parser.add_argument("--train", default=False, action='store_true')
    parser.add_argument("--num-epochs", type=int, default=30,
                        help="Number of epochs for training")
    parser.add_argument("--stop-time", type=str, default=None,
                        help="Number of seconds after which training on another epoch will not start."
                             "Hours are supported too, e.g. value '24h' should also work")
    parser.add_argument("--load-model_path", type=str, default=None,
                        help="Path to load the model from")
    parser.add_argument("--save-model-path", type=str, default="cross_encoder.pt",
                        help="Path to save the model to")
    parser.add_argument("--bert-model-name", type=str, default="FacebookAI/xlm-roberta-base",
                        help="Name of the BERT model")
    parser.add_argument("--train-data-path", type=str, default="DPR_pairs_train_50-60.json",
                        help="Train data filename jsonl, will be appended to 'data/DPR_pairs/'")
    parser.add_argument("--test-data-path", type=str, default="DPR_pairs_test.jsonl",
                        help="Test data filename jsonl, will be appended to 'data/DPR_pairs/'")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
                        help="Weight decay")
    parser.add_argument("--positive-weight", type=float, default=2,
                        help="Weight of positive class")
    parser.add_argument("--dropout-rate", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--label-smoothing", type=float, default=0,
                        help="Label smoothing rate (float between 0 and 1), default=0")
    parser.add_argument("--gradient-clip", type=float, default=None,
                        help="Gradient clipping `max_norm` param (float between 0 and 1), default=0")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--dont-save-model", default=False, action='store_true',
                        help="Don't save the model after training, can be useful for debugging")
    parser.add_argument("--evaluate-before-training", default=False, action='store_true',
                        help="Evaluate the model before training")
    parser.add_argument("--warmup-percent", type=float, default=0.1,
                        help="Warmup percent for the scheduler, default=0.1")

    parser.add_argument("--compute-recall-at-k", default=False, action='store_true')

    args = parser.parse_args()

    logs_suffix = "" if args.save_model_path == 'cross_encoder.pt' else args.save_model_path
    setup_logging(logs_suffix)

    # allows specifying train time in hours by converting hours to seconds
    # time in seconds is required in training loop
    if args.stop_time is not None and args.stop_time.endswith("h"):
        args.stop_time = int(args.stop_time.strip("h")) * 60 * 60

    if args.train:
        args.train_data_path = "data/DPR_pairs/" + args.train_data_path
        args.test_data_path = "data/DPR_pairs/" + args.test_data_path

        data_args = tce.TrainDataArgs(load_model_path=args.load_model_path,
                                      save_model_path=args.save_model_path,
                                      bert_model_name=args.bert_model_name,
                                      train_data_path=args.train_data_path,
                                      test_data_path=args.test_data_path,
                                      dont_save_model=args.dont_save_model)

        train_args = tce.TrainHyperparameters(num_epochs=args.num_epochs,
                                              lr=args.lr,
                                              weight_decay=args.weight_decay,
                                              positive_weight=args.positive_weight,
                                              dropout_rate=args.dropout_rate,
                                              stop_time=args.stop_time,
                                              label_smoothing=args.label_smoothing,
                                              gradient_clip=args.gradient_clip,
                                              batch_size=args.batch_size,
                                              evaluate_before_training=args.evaluate_before_training,
                                              )

        tce.train_ce(data_args, train_args)

    if args.compute_recall_at_k:
        ks = [1, 5, 10, 50, 200]

        paths = 'data/DPR_pairs/DPR_pairs_test.jsonl', 'data/DPR_pairs/DPR_pairs_validation.jsonl'
        for p in paths:
            print(f"Computing recall for {p}.")
            recalls = compute_recall(p, ks)
            for k, r in zip(ks, recalls):
                print(f"R@{k}: {r * 100:.2f}")
            print()
