import argparse

import train_ce as tce
from utils import compute_recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a cross encoder model.")

    parser.add_argument("--train", default=False, action='store_true')
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs for training")
    parser.add_argument("--stop_time", type=str, default=None,
                        help="Number of seconds after which training on another epoch will not start.")
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to load the model from")
    parser.add_argument("--save_model_path", type=str, default="cross_encoder.pt", help="Path to save the model to")
    parser.add_argument("--bert_model_name", type=str, default="FacebookAI/xlm-roberta-base",
                        help="Name of the BERT model")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-1, help="Weight decay")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")

    parser.add_argument("--compute_recall_at_k", default=False, action='store_true')

    args = parser.parse_args()

    # allows specifying train time in hours by converting hours to seconds
    # time in seconds is required in training loop
    if args.stop_time is not None and args.stop_time.endswith("h"):
        args.stop_time = int(args.stop_time.strip("h")) * 60 * 60

    if args.train:
        for key, value in vars(args).items():
            print(f"{key}: {value}")

        tce.train_ce(num_epochs=args.num_epochs,
                     load_model_path=args.load_model_path,
                     save_model_path=args.save_model_path,
                     bert_model_name=args.bert_model_name,
                     lr=args.lr,
                     weight_decay=args.weight_decay,
                     dropout_rate=args.dropout_rate,
                     stop_time=args.stop_time)

    if args.compute_recall_at_k:
        ks = [1, 5, 10, 50, 200]
        paths = 'data/DPR_pairs/DPR_pairs_test.jsonl', 'data/DPR_pairs/DPR_pairs_validation.jsonl'
        for p in paths:
            print(f"Computing recall for {p}.")
            recalls = compute_recall(p, ks)
            for k, r in zip(ks, recalls):
                print(f"R@{k}: {r * 100:.2f}")
            print()
