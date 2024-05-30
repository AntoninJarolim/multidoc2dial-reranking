import argparse

import train_ce as tce
from utils import compute_recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a cross encoder model.")

    parser.add_argument("--train", default=False, action='store_true')
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--nr_train_samples_testing", type=int, default=1000,
                        help="Number of training samples for testing")
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to load the model from")
    parser.add_argument("--save_model_path", type=str, default="cross_encoder.pt", help="Path to save the model to")
    parser.add_argument("--bert_model_name", type=str, default="FacebookAI/xlm-roberta-base",
                        help="Name of the BERT model")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-1, help="Weight decay")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")

    parser.add_argument("--compute_recall_at_k", default=False, action='store_true')

    args = parser.parse_args()

    if args.train:
        print(f"num_epochs: {args.num_epochs}")
        print(f"nr_train_samples_testing: {args.nr_train_samples_testing}")
        print(f"load_model_path: {args.load_model_path}")
        print(f"save_model_path: {args.save_model_path}")
        print(f"bert_model_name: {args.bert_model_name}")
        print(f"lr: {args.lr}")
        print(f"weight_decay: {args.weight_decay}")
        print(f"dropout_rate: {args.dropout_rate}")

        tce.train_ce(num_epochs=args.num_epochs,
                     nr_train_samples_testing=args.nr_train_samples_testing,
                     load_model_path=args.load_model_path,
                     save_model_path=args.save_model_path,
                     bert_model_name=args.bert_model_name,
                     lr=args.lr,
                     weight_decay=args.weight_decay,
                     dropout_rate=args.dropout_rate)

    if args.compute_recall_at_k:
        ks = [1, 5, 10, 50, 200]
        paths = 'data/DPR_pairs/DPR_pairs_test.jsonl', 'data/DPR_pairs/DPR_pairs_validation.jsonl'
        for p in paths:
            print(f"Computing recall for {p}.")
            recalls = compute_recall(p, ks)
            for k, r in zip(ks, recalls):
                print(f"R@{k}: {r * 100:.2f}")
            print()