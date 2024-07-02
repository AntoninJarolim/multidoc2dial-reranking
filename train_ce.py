import json
import logging
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from md2d_dataset import MD2DDataset
from utils import mrr_metric, transform_batch, pred_recall_metric

logger = logging.getLogger('main')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"device: {device}")
start_t = time.time()

tb_writer = SummaryWriter()


@dataclass
class EvaluationMetrics:
    average_mrr: float
    sum_recalls: torch.Tensor
    loss_total: float
    loss_negative: float
    loss_positive: float
    softmax_loss_sum: float

    def log(self, epoch, dataset_name, gs):
        logger.info(f"({time.time() - start_t:0.2f}) Epoch {epoch}; gs:{gs}; on {dataset_name} dataset: "
                    f"MRR: {self.average_mrr:0.2f}; "
                    f"Recalls: {self.sum_recalls}; "
                    f"Loss: {self.loss_total:0.2f}; "
                    f"positive-loss: {self.loss_positive:0.2f}; "
                    f"negative-loss: {self.loss_negative:0.2f}; "
                    f"softmax-loss: {self.softmax_loss_sum:0.2f}; ")

        # tb_writer.add_scalars(f"{dataset_name}/Metrics", {
        #     "MRR": self.average_mrr,
        #     "Recall@1": self.sum_recalls[0],
        #     "Recall@5": self.sum_recalls[1],
        #     "Recall@10": self.sum_recalls[2]}, gs)
        tb_writer.add_scalar(f"{dataset_name}/MRR", self.average_mrr, gs)
        tb_writer.add_scalar(f"{dataset_name}/Recall@1", self.sum_recalls[0], gs)
        tb_writer.add_scalar(f"{dataset_name}/Recall@5", self.sum_recalls[1], gs)
        tb_writer.add_scalar(f"{dataset_name}/Recall@10", self.sum_recalls[2], gs)

        tb_writer.add_scalars(f"{dataset_name}/Losses", {
            "positive": self.loss_positive,
            "negative": self.loss_negative,
            "total": self.loss_total}, gs)

        tb_writer.add_scalar(f"{dataset_name}/LossTotal", self.loss_total, gs)
        tb_writer.add_scalar(f"{dataset_name}/LossSoftMax", self.softmax_loss_sum, gs)


class CrossEncoder(torch.nn.Module):
    def __init__(self, bert_model_name, dropout_rate=0.1):
        super(CrossEncoder, self).__init__()
        m = AutoModelForSequenceClassification.from_pretrained(bert_model_name)
        self.bert_model = m.bert

        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(384, 1)
        # Try loading linear layer from model
        try:
            self.linear.weight = m.classifier.weight
            self.linear.bias = m.classifier.bias
        except Exception:
            pass
            # init.xavier_uniform_(self.linear.weight)
            # if self.linear.bias is not None:
            #     init.zeros_(self.linear.bias)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        cls_features = self.get_cls_features(input_ids, attention_mask, token_type_ids)
        cls_features = self.dropout(cls_features)
        logits = self.linear(cls_features)
        # Flatten [batch_size, 1] to [batch_size]
        return logits.flatten()

    def get_cls_features(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # The last hidden state
        last_hidden_state = hidden_states[-1]
        # The [CLS] token is the first token of the last hidden state
        cls_features = last_hidden_state[:, 0, :]
        return outputs[1]

    @torch.no_grad()
    def evaluate(self, data_loader, loss_fn, take_n=0, save_all_losses=True) -> EvaluationMetrics:
        rr_total = 0
        loss_negative, loss_positive = 0, 0
        recalls = []

        # Required for save_all_losses=True
        bce_no_reduce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1), reduction='none')
        losses = torch.empty(0)
        labels = torch.empty(0)

        # Try to also compute softmax loss
        softmax = nn.Softmax(dim=0)
        bce_loss = nn.BCELoss()
        sm_loss_sum = 0

        for i, batch in enumerate(data_loader):
            batch = transform_batch(batch, take_n)
            batch = {k: v.to(device) for k, v in batch.items()}
            if take_n == 0:
                pred = self.process_large_batch(batch, 64)
            else:
                pred = self(input_ids=batch['in_ids'],
                            attention_mask=batch['att_mask'],
                            token_type_ids=batch['tt_ids'])
            loss_negative, loss_positive = separate_losses(batch, loss_fn, pred, loss_negative, loss_positive)

            sm_loss_sum += bce_loss(softmax(pred), batch['label']).cpu()
            # Compute MRR for the current batch
            batch_mrr = mrr_metric(pred, batch['label'])
            rr_total += batch_mrr
            # Compute R@ks for the current batch
            recall = pred_recall_metric(pred, batch['label'], [1, 5, 10])
            recalls.append(recall)

            if save_all_losses:
                loss = bce_no_reduce(pred, batch['label']).cpu()
                losses = torch.cat((losses, loss), dim=0)
                labels = torch.cat((labels, batch['label'].cpu()), dim=0)

        if save_all_losses:
            all_data = {
                'labels': labels.tolist(),
                'losses': losses.tolist()
            }
            json.dump(all_data, open("validation_losses_ls1.json", mode="w"))

        recalls = torch.tensor(recalls)
        N = recalls.shape[0]
        sum_recalls = torch.sum(recalls, dim=0) / N

        num_batches = len(data_loader)
        average_mrr = rr_total / num_batches if num_batches > 0 else 0

        loss_total = loss_negative + loss_positive
        return EvaluationMetrics(average_mrr, sum_recalls, loss_total, loss_negative, loss_positive, sm_loss_sum)

    def process_large_batch(self, batch, max_sub_batch_size):
        # Split the batch into smaller sub-batches
        input_ids = batch['in_ids']
        attention_mask = batch['att_mask']
        token_type_ids = batch['tt_ids']

        num_samples = input_ids.size(0)

        preds = []
        for i in range(0, num_samples, max_sub_batch_size):
            sub_pred = self(input_ids=input_ids[i:i + max_sub_batch_size],
                            attention_mask=attention_mask[i:i + max_sub_batch_size],
                            token_type_ids=token_type_ids[i:i + max_sub_batch_size])
            preds.append(sub_pred)

        # Merge predictions back
        merged_preds = torch.cat(preds, dim=0)
        return merged_preds


def load_model(cross_encoder, load_path):
    cross_encoder.load_state_dict(torch.load(load_path))
    logger.info(f"Model loaded successfully from {load_path}")


def train_ce(num_epochs=30,
             load_model_path=None,
             save_model_path="cross_encoder.pt",
             bert_model_name="FacebookAI/xlm-roberta-base",
             train_data_path='data/DPR_pairs/DPR_pairs_train.jsonl',
             test_data_path='data/DPR_pairs/DPR_pairs_test.jsonl',
             lr=1e-5,
             weight_decay=1e-1,
             dropout_rate=0.1,
             optimizer=None,
             loss_fn=None,
             stop_time=None,
             label_smoothing=0,
             gradient_clip=0,
             batch_size=128
             ):
    gradient_clip = None if gradient_clip == 0 else gradient_clip

    cross_encoder = CrossEncoder(bert_model_name, dropout_rate=dropout_rate)
    if load_model_path is not None:
        load_model(cross_encoder, load_model_path)
    cross_encoder.to(device)

    optimizer = optimizer or AdamW(cross_encoder.parameters(), lr=lr)  # , weight_decay=weight_decay)
    loss_fn = loss_fn or nn.BCEWithLogitsLoss()  # pos_weight=torch.tensor(2))

    try:
        pass
        training_loop(cross_encoder, loss_fn, num_epochs, optimizer, bert_model_name, stop_time, label_smoothing,
                      gradient_clip, train_data_path, batch_size, test_data_path)
    except KeyboardInterrupt:
        logger.info(f"Early stopping by user Ctrl+C interaction.")

    torch.save(cross_encoder.state_dict(), save_model_path)
    logger.info(f"Model saved to {save_model_path}")

    cross_encoder.eval()
    with torch.no_grad():
        test_dataset = MD2DDataset(test_data_path,
                                   bert_model_name)
        test_loader = DataLoader(test_dataset, batch_size=1)
        evaluation = cross_encoder.evaluate(test_loader,
                                            loss_fn,
                                            save_all_losses=True)
        # evaluation.log(num_epochs, "test", 0)


def training_loop(cross_encoder, loss_fn, num_epochs, optimizer, bert_model_name, stop_time, label_smoothing,
                  gradient_clip, train_data_path, batch_size, test_data_path):
    train_dataset = MD2DDataset(train_data_path,
                                bert_model_name,
                                label_smoothing=label_smoothing,
                                shuffle=False)

    test_dataset = MD2DDataset(test_data_path,
                               bert_model_name)
    # val_dataset = MD2DDataset('data/DPR_pairs/DPR_pairs_validation.jsonl',
    #                           bert_model_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    # only for evaluation - there are 10 negative samples and 1 positive sample
    test_batch_size = 1
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
    # val_loader = DataLoader(val_dataset, batch_size=test_batch_size)

    # TEST_EVERY = len(train_loader)
    TEST_EVERY = 500
    logger.info(f"TEST_EVERY: {TEST_EVERY}")
    total_loss, loss_positive, loss_negative, gradient_steps = 0, 0, 0, 0
    for epoch in range(num_epochs):
        logger.info(f"({time.time() - start_t:0.2f})------------- Training epoch {epoch} started -------------")

        for i, batch in enumerate(train_loader):
            # TESTING each x gradient steps
            if gradient_steps % TEST_EVERY == 0:
                cross_encoder.eval()
                with torch.no_grad():
                    evaluation = cross_encoder.evaluate(test_loader, loss_fn,
                                                        save_all_losses=False)
                    evaluation.log(epoch, "test", gradient_steps)
                cross_encoder.train()

            # Training step
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            pred = cross_encoder(input_ids=batch['in_ids'], attention_mask=batch['att_mask'])
            loss = loss_fn(pred, batch['label'])

            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(cross_encoder.parameters(), gradient_clip)
            optimizer.step()
            total_loss += loss.item()

            loss_negative, loss_positive = separate_losses(batch, loss_fn, pred, loss_negative, loss_positive)

            gradient_steps += 1

        # logging train losses after each epoch
        logger.info(f"({time.time() - start_t:0.2f}) "
                    f"Epoch: {epoch}; gs: {gradient_steps}; on training dataset: "
                    f"loss: {loss_positive + loss_negative}; "
                    f"positive-loss: {loss_positive:0.2f}; "
                    f"negative-loss: {loss_negative:0.2f} ")

        tb_writer.add_scalar("train/TotalLoss", total_loss, gradient_steps)
        tb_writer.flush()
        total_loss, loss_positive, loss_negative = 0, 0, 0

        # stopping based on max time to train
        if stop_time is not None and time.time() - start_t > stop_time:
            break

    tb_writer.close()


def separate_loss(batch, loss_fn, pred, mask_label):
    if mask_label == 0:
        mask = (batch['label'] < 0.5).flatten()
    elif mask_label == 1:
        mask = (batch['label'] >= 0.5).flatten()
    else:
        raise AssertionError("mask_label must be either 0 or 1.")

    mask_loss = 0
    if mask.sum() > 0:  # Check if there are samples
        pred_mask = pred[mask]
        mask_labels = batch['label'][mask]
        mask_loss = loss_fn(pred_mask, mask_labels).item()
    return mask_loss


def separate_losses(batch, loss_fn, pred, loss_negative, loss_positive):
    loss_positive += separate_loss(batch, loss_fn, pred, 1)
    loss_negative += separate_loss(batch, loss_fn, pred, 0)
    return loss_negative, loss_positive


if __name__ == "__main__":
    import torch

    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')

    features = tokenizer(['How many people live in Berlin?', 'How many people live in Berlin?'], [
        'Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.',
        'New York City is famous for the Metropolitan Museum of Art.'], padding=True, truncation=True,
                         return_tensors="pt")

    bert_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    cross_encoder = CrossEncoder(bert_model_name)

    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
        print(scores)

    # train_ce()
