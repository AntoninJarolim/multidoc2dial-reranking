import logging
import time

import torch
import torch.nn.init as init
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM

from md2d_dataset import MD2DDataset
from utils import mrr_metric, transform_batch, pred_recall_metric

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_t = time.time()


class CrossEncoder(torch.nn.Module):
    def __init__(self, bert_model_name, dropout_rate=0.1):
        super(CrossEncoder, self).__init__()

        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_model_name)
        self.linear = nn.Linear(768, 1)
        self.dropout = nn.Dropout(dropout_rate)

        init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, input_ids, attention_mask):
        cls_features = self.get_cls_features(input_ids, attention_mask)
        cls_features = self.dropout(cls_features)
        logits = self.linear(cls_features)
        # Flatten [batch_size, 1] to [batch_size]
        return logits.flatten()

    def get_cls_features(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # The last hidden state
        last_hidden_state = hidden_states[-1]
        # The [CLS] token is the first token of the last hidden state
        cls_features = last_hidden_state[:, 0, :]
        return cls_features

    def evaluate(self, data_loader, loss_fn, take_n=0):
        rr_total = 0
        loss_negative, loss_positive = 0, 0
        recalls = []

        for i, batch in enumerate(data_loader):
            batch = transform_batch(batch, take_n)
            batch = {k: v.to(device) for k, v in batch.items()}
            if take_n == 0:
                pred = self.process_large_batch(batch, 40)
            else:
                pred = self(input_ids=batch['in_ids'], attention_mask=batch['att_mask'])
            loss_negative, loss_positive = separate_losses(batch, loss_fn, pred, loss_negative, loss_positive)
            # Compute MRR for the current batch
            batch_mrr = mrr_metric(pred, batch['label'])
            rr_total += batch_mrr
            # Compute R@ks for the current batch
            recall = pred_recall_metric(pred, batch['label'], [1, 5, 10])
            recalls.append(recall)

        recalls = torch.tensor(recalls)
        N = recalls.shape[0]
        sum_recalls = torch.sum(recalls, dim=0) / N

        num_batches = len(data_loader)
        average_mrr = rr_total / num_batches if num_batches > 0 else 0

        loss_total = loss_negative + loss_positive
        return average_mrr, sum_recalls, loss_total, loss_negative, loss_positive

    def process_large_batch(self, batch, max_sub_batch_size):
        # Split the batch into smaller sub-batches
        input_ids = batch['in_ids']
        attention_mask = batch['att_mask']

        num_samples = input_ids.size(0)
        sub_batches = [(input_ids[i:i + max_sub_batch_size], attention_mask[i:i + max_sub_batch_size])
                       for i in range(0, num_samples, max_sub_batch_size)]

        preds = []
        for sub_input_ids, sub_attention_mask in sub_batches:
            sub_input_ids = sub_input_ids.to(device)
            sub_attention_mask = sub_attention_mask.to(device)
            sub_pred = self(input_ids=sub_input_ids, attention_mask=sub_attention_mask)
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
             lr=1e-5,
             weight_decay=1e-1,
             dropout_rate=0.1,
             optimizer=None,
             loss_fn=None,
             stop_time=None):
    cross_encoder = CrossEncoder(bert_model_name, dropout_rate=dropout_rate)
    if load_model_path is not None:
        load_model(cross_encoder, load_model_path)
    cross_encoder.to(device)

    optimizer = optimizer or AdamW(cross_encoder.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = loss_fn or nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10))

    try:
        training_loop(cross_encoder, loss_fn, num_epochs, optimizer, bert_model_name, stop_time)
    except KeyboardInterrupt:
        logger.info(f"Early stopping by user Ctrl+C interaction.")

    torch.save(cross_encoder.state_dict(), save_model_path)
    logger.info(f"Model saved to {save_model_path}")

    cross_encoder.eval()
    with torch.no_grad():
        test_dataset = MD2DDataset('data/DPR_pairs/DPR_pairs_test.jsonl',
                                   bert_model_name)
        test_loader = DataLoader(test_dataset, batch_size=1)
        average_mrr, sum_recalls, total_loss, pos_loss, neg_loss = cross_encoder.evaluate(test_loader, loss_fn)
        logger.info(f"({time.time() - start_t:0.2f}) Final test on test dataset!"
                    f"MRR: {average_mrr:0.2f}; "
                    f"Recalls: {sum_recalls} "
                    f"Loss: {total_loss:0.2f}"
                    f"positive-loss: {pos_loss:0.2f}; "
                    f"negative-loss: {neg_loss:0.2f} ")


def training_loop(cross_encoder, loss_fn, num_epochs, optimizer, bert_model_name, stop_time):
    train_dataset = MD2DDataset('data/DPR_pairs/DPR_pairs_train.jsonl',
                                bert_model_name,
                                shuffle=False)

    test_dataset = MD2DDataset('data/DPR_pairs/DPR_pairs_test.jsonl',
                               bert_model_name)
    # val_dataset = MD2DDataset('data/DPR_pairs/DPR_pairs_validation.jsonl',
    #                           bert_model_name)

    batch_size = 33
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    # only for evaluation - there are 10 negative samples and 1 positive sample
    test_batch_size = 1
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
    # val_loader = DataLoader(val_dataset, batch_size=test_batch_size)

    for epoch in range(num_epochs):
        logger.info(f"({time.time() - start_t:0.2f})------------- Training epoch {epoch} started -------------")

        # TESTING - MRR
        cross_encoder.eval()
        with torch.no_grad():
            average_mrr, sum_recalls, total_loss, pos_loss, neg_loss = cross_encoder.evaluate(test_loader, loss_fn,
                                                                                              take_n=32)
            logger.info(f"({time.time() - start_t:0.2f}) Epoch {epoch} on test dataset: "
                        f"MRR: {average_mrr:0.2f}; "
                        f"Recalls: {sum_recalls} "
                        f"Loss: {total_loss:0.2f}"
                        f"positive-loss: {pos_loss:0.2f}; "
                        f"negative-loss: {neg_loss:0.2f} ")

        # training
        total_loss, loss_positive, loss_negative = 0, 0, 0
        cross_encoder.train()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            pred = cross_encoder(input_ids=batch['in_ids'], attention_mask=batch['att_mask'])
            loss = loss_fn(pred, batch['label'])

            loss.backward()
            optimizer.step()
            train_batch_loss = loss.item()

            if i % 500 == 0:
                logger.info(
                    f'({time.time() - start_t:0.2f}) Training batch {i}/{len(train_loader)} loss: {train_batch_loss}')
            total_loss += train_batch_loss

            loss_negative, loss_positive = separate_losses(batch, loss_fn, pred, loss_negative, loss_positive)

        logger.info(f"({time.time() - start_t:0.2f}) Epoch {epoch} on training dataset: "
                    f"loss: {loss_positive + loss_negative}; "
                    f"positive-loss: {loss_positive:0.2f}; "
                    f"negative-loss: {loss_negative:0.2f} ")

        if stop_time is not None and time.time() - start_t > stop_time:
            break


def separate_loss(batch, loss_fn, pred, mask_label):
    mask = (batch['label'] == mask_label).flatten()
    mask_loss = 0
    if mask.sum() > 0:  # Check if there are samples
        pred_positive = pred[mask]
        mask_labels = batch['label'][mask]
        mask_loss = loss_fn(pred_positive, mask_labels).item()
    return mask_loss


def separate_losses(batch, loss_fn, pred, loss_negative, loss_positive):
    loss_positive += separate_loss(batch, loss_fn, pred, 1)
    loss_negative += separate_loss(batch, loss_fn, pred, 0)
    return loss_negative, loss_positive


if __name__ == "__main__":
    train_ce()
