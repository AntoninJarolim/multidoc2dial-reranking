# set logger to info mode
import logging

import jsonlines
import torch
from sentence_transformers import InputExample, CrossEncoder
from torch.utils.data import DataLoader, Dataset

from utils import pred_recall_metric, mrr_metric

logging.basicConfig(level=logging.INFO)


# Define a custom PyTorch dataset
class Md2dSTDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_and_process_jsonlines(file_path):
    train_samples = []

    with jsonlines.open(file_path) as reader:
        for obj in reader:
            x = obj['x']
            label = obj['label']

            # Split the text by [SEP] token
            sentences = x.split('[SEP]')
            if len(sentences) != 2:
                print("Incorrect input format.")
                exit(-1)  # Skip if the format is not as expected

            sentence1, sentence2 = sentences
            train_samples.append(InputExample(texts=[sentence1.strip(), sentence2.strip()],
                                              label=int(label)))

    return train_samples


def load_and_process_jsonlines_eval(dev_file_path):
    """
    samples (List[Dict, str, Union[str, List[str]]) – Must be a list and each element is of the form:
    {‘query’: ‘’, ‘positive’: [], ‘negative’: []}. Query is the search query, positive is a list
    of positive (relevant) documents, negative is a list of negative (irrelevant) documents.
    """
    dev_samples = []

    with jsonlines.open(dev_file_path) as reader:
        for example_line in reader:
            query = None
            positive = []
            negative = []

            for example in example_line:
                cur_query, cur_doc = example['x'].split('[SEP]')
                if query is None:
                    query = cur_query
                else:
                    assert cur_query == query, "Incorrect input format."

                if example['label'] == 1:
                    positive.append(cur_doc)
                else:
                    negative.append(cur_doc)

            dev_samples.append({
                'query': query,
                'positive': positive,
                'negative': negative
            })

    return dev_samples


import csv
import logging
import os
from typing import Optional

import numpy as np
from sklearn.metrics import ndcg_score

logger = logging.getLogger(__name__)


class CERerankingEvaluator:
    """
    This class evaluates a CrossEncoder model for the task of re-ranking.

    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10 and NDCG@10 are computed to measure the quality of the ranking.

    Args:
        samples (List[Dict, str, Union[str, List[str]]): Must be a list and each element is of the form:
            {'query': '', 'positive': [], 'negative': []}. Query is the search query, positive is a list
            of positive (relevant) documents, negative is a list of negative (irrelevant) documents.
    """

    def __init__(
            self, samples, at_k: int = 10, name: str = "", write_csv: bool = True, mrr_at_k: Optional[int] = None
    ):
        self.samples = samples
        self.name = name
        if mrr_at_k is not None:
            logger.warning(f"The `mrr_at_k` parameter has been deprecated; please use `at_k={mrr_at_k}` instead.")
            self.at_k = mrr_at_k
        else:
            self.at_k = at_k

        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())

        self.csv_file = "CERerankingEvaluator" + ("_" + name if name else "") + f"_results_@{self.at_k}.csv"
        self.csv_headers = [
            "epoch",
            "steps",
            "MRR@{}".format(self.at_k),
            "NDCG@{}".format(self.at_k),
        ]
        self.write_csv = write_csv

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("CERerankingEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        all_mrr_scores = []
        all_mrr_scores_fado = []
        all_ndcg_scores = []
        num_queries = 0
        num_positives = []
        num_negatives = []
        recalls = []
        for instance in self.samples:
            query = instance["query"]
            positive = list(instance["positive"])
            negative = list(instance["negative"])
            docs = positive + negative
            is_relevant = [1] * len(positive) + [0] * len(negative)

            if len(positive) == 0 or len(negative) == 0:
                continue

            num_queries += 1
            num_positives.append(len(positive))
            num_negatives.append(len(negative))

            model_input = [[query, doc] for doc in docs]
            pred_scores = model.predict(model_input, convert_to_numpy=True, show_progress_bar=False)
            pred_scores_argsort = np.argsort(-pred_scores)  # Sort in decreasing order

            mrr_score = 0
            for rank, index in enumerate(pred_scores_argsort[0: self.at_k]):
                if is_relevant[index]:
                    mrr_score = 1 / (rank + 1)
                    break

            all_mrr_scores.append(mrr_score)
            all_ndcg_scores.append(ndcg_score([is_relevant], [pred_scores], k=self.at_k))

            # Compute R@ks for the current batch
            recall = pred_recall_metric(pred_scores, is_relevant, [1, 5, 10])
            recalls.append(recall)

            fado_mrr = mrr_metric(pred_scores, is_relevant)
            all_mrr_scores_fado.append(fado_mrr)

        mean_fado_mrr = np.mean(all_mrr_scores_fado)
        mean_mrr = np.mean(all_mrr_scores)
        mean_ndcg = np.mean(all_ndcg_scores)

        recalls = torch.tensor(recalls)
        N = recalls.shape[0]
        sum_recalls = torch.sum(recalls, dim=0) / N

        logger.info(
            "Queries: {} \t Positives: Min {:.1f}, Mean {:.1f}, Max {:.1f} \t Negatives: Min {:.1f}, Mean {:.1f}, Max {:.1f}".format(
                num_queries,
                np.min(num_positives),
                np.mean(num_positives),
                np.max(num_positives),
                np.min(num_negatives),
                np.mean(num_negatives),
                np.max(num_negatives),
            )
        )
        logger.info("MRR@{}: {:.2f}".format(self.at_k, mean_mrr * 100))
        logger.info("FADO MRR: {:.2f}".format(mean_fado_mrr * 100))
        logger.info("NDCG@{}: {:.2f}".format(self.at_k, mean_ndcg * 100))
        logger.info("Recall@1: {:.2f}".format(sum_recalls[0] * 100))
        logger.info("Recall@5: {:.2f}".format(sum_recalls[1] * 100))
        logger.info("Recall@10: {:.2f}".format(sum_recalls[2] * 100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mean_mrr, mean_ndcg])

        return mean_mrr


# Example usage
file_path = 'data/DPR_pairs/DPR_pairs_train_50-60.json'
train_samples = load_and_process_jsonlines(file_path)

st_ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', num_labels=1)

train_dataset = Md2dSTDataset(train_samples)
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

# Load the dev set
dev_file_path = 'data/DPR_pairs/DPR_pairs_test.jsonl'
dev_samples = load_and_process_jsonlines_eval(dev_file_path)
evaluator = CERerankingEvaluator(dev_samples, name="MD2D-dev", at_k=200)

mrr = evaluator(st_ce_model)
print(f"Mean Reciprocal Rank (MRR) on MD2D-dev: {mrr:.4f}")

num_epochs = 3
st_ce_model.fit(
    train_dataloader=train_dataloader,
    epochs=num_epochs,
    save_best_model=False,
    show_progress_bar=False,
    evaluator=evaluator
)

mrr = evaluator(st_ce_model)
print(f"Mean Reciprocal Rank after training (MRR) on MD2D-dev: {mrr:.4f}")
