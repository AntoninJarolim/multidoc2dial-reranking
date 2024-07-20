import json
import logging
import socket

import numpy as np
import torch

logger = logging.getLogger('main')


def conv_to_torch(pred, labels):
    if type(pred) is not torch.Tensor:
        pred = torch.tensor(pred)
    if type(labels) is not torch.Tensor:
        labels = torch.tensor(labels)
    return pred, labels


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

    preds, labels = conv_to_torch(preds, labels)
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
    pred, labels = conv_to_torch(pred, labels)

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
    tt_ids = torch.vstack([item['tt_ids'] for item in batch])

    # Combine into a dictionary as expected by your code
    return {
        'label': labels,
        'in_ids': in_ids,
        'att_mask': att_masks,
        'tt_ids': tt_ids
    }


def calc_physical_batch_size(batch_size, gpu_batches=None):
    """
    Calculates logical and physical batch size based on the GPU configuration.
    :param batch_size: Actual batch size
    :param gpu_batches: Json dict with GPU name as key and physical batch size as value
    :return: Tuple of logical and physical batch size
    """
    assert batch_size > 0, "Batch size should greater than 0"

    default_gpu_batches = {
        "NVIDIA GeForce RTX 2080 Ti": 1,
        "NVIDIA GeForce RTX 3090": 4,
        "NVIDIA RTX A5000": 4,
    }
    gpu_batches = gpu_batches or default_gpu_batches

    # Get the host name
    host_name = socket.gethostname()
    hosts_gpu_file = ".host_config.json"  # Json dict with host name as key and gpu name as value
    gpu_hosts = json.load(open(hosts_gpu_file))
    assert host_name in gpu_hosts, f"Host {host_name} not found in {hosts_gpu_file} file"

    # Calculate logical and physical batch size
    train_batch_size = gpu_batches[gpu_hosts[host_name]]
    gradient_accumulation_steps = batch_size // train_batch_size

    # Do not accumulate gradients if the batch size is less than the physical batch size
    if train_batch_size >= batch_size:
        return batch_size, 1

    return train_batch_size, gradient_accumulation_steps


def load_model(cross_encoder, load_path):
    cross_encoder.load_state_dict(torch.load(load_path))
    logger.info(f"Model loaded successfully from {load_path}")
    return cross_encoder


def save_model(cross_encoder, save_model_path, msg_str=None):
    torch.save(cross_encoder.state_dict(), save_model_path)
    msg_str = msg_str if msg_str is not None else "Model saved to "
    logger.info(f"{msg_str} {save_model_path}")


def save_best_model(cross_encoder, save_model_path):
    save_model(cross_encoder, save_model_path, "New best saved to ")


from torch.utils.data import DataLoader, Dataset
import itertools

class LimitedDataLoader:
    def __init__(self, dataloader, max_iterations):
        self.dataloader = dataloader
        self.max_iterations = max_iterations

    def __iter__(self):
        self.iter_loader = iter(self.dataloader)
        self.counter = 0
        return self

    def __next__(self):
        if self.counter < self.max_iterations:
            self.counter += 1
            return next(self.iter_loader)
        else:
            raise StopIteration

    def __len__(self):
        return min(self.max_iterations, len(self.dataloader))


