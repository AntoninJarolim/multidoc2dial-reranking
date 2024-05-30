import json

import torch


def mrr_metric(preds, labels):
    """
    Compute Mean Reciprocal Rank (MRR).

    Arguments:
    pred -- Tensor of predicted scores.
    labels -- Tensor of true labels.

    Returns:
    MRR -- Mean Reciprocal Rank for the batch.
    """
    # Ensure labels are in the correct shape
    if not (labels[0] == 1 and all(label == 0 for label in labels[1:])):
        if len(labels) % 11 != 0:
            raise AssertionError("First label must be target and others non-targets")

        # reshape and call recursively if not
        # e.g. when preds and labels has sizes 33, it reshapes it to (3, 11) and calls recursively
        nr_splits = len(labels) // 11
        # take 11 members from a multiply by 11
        preds_split = torch.reshape(preds, (nr_splits, 11))
        labels_split = torch.reshape(labels, (nr_splits, 11))
        total_rank, total_rr = 0, 0
        for x in range(nr_splits):
            rr, rank = mrr_metric(preds_split[x], labels_split[x])
            total_rr += rr
            total_rank += rank
        return total_rr / nr_splits, total_rank / nr_splits

    # Get the indices that would sort the predictions in descending order
    sorted_indices = torch.argsort(preds, descending=True)

    # Find the rank of the true target (which is always at index 0 in labels)
    rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1

    # Calculate the reciprocal rank
    return 1.0 / rank, rank


def recall_metric(sorted_labels, ks):
    if type(ks) is not list:
        if type(ks) is int:
            ks = [ks]
        else:
            raise AssertionError("Type of ks parameter has to be list or int.")

    recall_ks = [sum(sorted_labels[:k]) for k in ks]
    return recall_ks


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
