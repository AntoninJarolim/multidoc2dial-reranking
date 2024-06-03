import json
import logging

import numpy as np
import torch

logger = logging.getLogger('main')


def mrr_metric(preds, labels):
    """
    Compute Mean Reciprocal Rank (MRR).

    Arguments:
    pred -- Tensor of predicted scores.
    labels -- Tensor of true labels.

    Returns:
    MRR -- Mean Reciprocal Rank for the batch.
    """

    # skip computation if all labels are 0
    if sum(labels) == 0:
        return 0

    sorted_indices = torch.argsort(preds, descending=True)
    sorted_labels = labels[sorted_indices]
    rank = np.argwhere(sorted_labels.cpu()).flatten().item() + 1
    # Calculate the reciprocal rank
    return 1.0 / rank


def recall_metric(sorted_labels, ks):
    if type(ks) is not list:
        if type(ks) is int:
            ks = [ks]
        else:
            raise AssertionError("Type of ks parameter has to be list or int.")

    recall_ks = [sum(sorted_labels[:k]) for k in ks]
    return recall_ks


def pred_recall_metric(pred, labels, ks):
    # skip computation if all labels are 0
    if sum(labels) == 0:
        return torch.zeros(len(ks))

    sorted_indexes = torch.argsort(pred, descending=True)
    sorted_labels = labels[sorted_indexes]
    return recall_metric(sorted_labels, ks)


def compute_recall(path, ks):
    with open(path, 'r') as file:
        recalls = []
        for line in file:
            dpr_pairs = json.loads(line.strip())
            sorted_labels = [int(x['label']) for x in dpr_pairs]
            recall_ks = recall_metric(sorted_labels, ks)
            recalls.append(recall_ks)

        recalls = torch.tensor(recalls)
        N = recalls.shape[0]
        sum_recalls = torch.sum(recalls, dim=0) / N
    return sum_recalls


def transform_batch(batch, take_n=0):
    if take_n > 0:
        batch = batch[:take_n]
    # Gather 'label', 'in_ids', and 'att_mask' from these members
    labels = torch.vstack([item['label'] for item in batch]).flatten()
    in_ids = torch.vstack([item['in_ids'] for item in batch])
    att_masks = torch.vstack([item['att_mask'] for item in batch])

    # Combine into a dictionary as expected by your code
    return {
        'label': labels,
        'in_ids': in_ids,
        'att_mask': att_masks
    }
