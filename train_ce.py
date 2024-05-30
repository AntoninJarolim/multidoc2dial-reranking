import logging
import time

import torch
import torch.nn.init as init
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM

from md2d_dataset import MD2DDataset
from utils import mrr_metric

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start = time.time()


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

    def evaluate(self, data_loader, loss_fn, stop_after=None):
        rr_total = 0
        rank_total = 0
        loss_total = 0

        for i, batch in enumerate(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = self(input_ids=batch['in_ids'], attention_mask=batch['att_mask'])
            loss = loss_fn(pred, batch['label'])
            loss_total += loss.item()
            # Compute MRR for the current batch
            batch_mrr, rank = mrr_metric(pred, batch['label'])
            rr_total += batch_mrr
            rank_total += rank

            if stop_after and i > stop_after:
                break

        num_batches = stop_after if stop_after is not None else len(data_loader)
        average_mrr = rr_total / num_batches if num_batches > 0 else 0
        average_rank = rank_total / num_batches if num_batches > 0 else 0
        return average_mrr, average_rank, loss_total


def load_model(cross_encoder, model_path):
    load_path = "train_ce_e5step_dropout.pt"
    cross_encoder.load_state_dict(torch.load(load_path))
    logger.info(f"Model loaded successfully from {load_path}")


def train_ce(num_epochs=100,
             nr_train_samples_testing=1000,
             load_model_path=None,
             save_model_path="cross_encoder.pt",
             bert_model_name="FacebookAI/xlm-roberta-base",
             lr=1e-5,
             weight_decay=1e-1,
             dropout_rate=0.1,
             optimizer=None,
             loss_fn=None):
    cross_encoder = CrossEncoder(bert_model_name, dropout_rate=dropout_rate)
    if load_model_path is not None:
        load_model(cross_encoder, load_model_path)
    cross_encoder.to(device)

    optimizer = optimizer or AdamW(cross_encoder.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = loss_fn or nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10))

    try:
        training_loop(cross_encoder, loss_fn, nr_train_samples_testing, num_epochs, optimizer, bert_model_name)
    except KeyboardInterrupt:
        logger.info(f"Early stopping by user Ctrl+C interaction.")

    torch.save(cross_encoder.state_dict(), save_model_path)
    logger.info(f"Model saved to {save_model_path}")


def training_loop(cross_encoder, loss_fn, nr_train_samples_testing, num_epochs, optimizer, bert_model_name):
    train_dataset = MD2DDataset('data/DPR_pairs/DPR_pairs_train.jsonl',
                                bert_model_name,
                                shuffle=False)

    test_dataset = MD2DDataset('data/DPR_pairs/DPR_pairs_test.jsonl',
                               bert_model_name)
    val_dataset = MD2DDataset('data/DPR_pairs/DPR_pairs_validation.jsonl',
                              bert_model_name)

    batch_size = 33
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    # only for evaluation - there are 10 negative samples and 1 positive sample
    test_batch_size = 11
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size)

    for epoch in range(num_epochs):
        logger.info(f"({time.time() - start:0.2f}) Epoch: {epoch}")

        # TESTING - MRR
        cross_encoder.eval()
        with torch.no_grad():
            average_mrr, average_rank, total_loss = cross_encoder.evaluate(test_loader, loss_fn)
            logger.info(f"({time.time() - start:0.2f}) Epoch {epoch} on test dataset: "
                        f"MRR: {average_mrr:0.4f}; "
                        f"AvgRank: {average_rank:0.4f} "
                        f"Loss: {total_loss}")

            nr_samples = nr_train_samples_testing
            average_mrr, average_rank, total_loss = cross_encoder.evaluate(train_loader,
                                                                           loss_fn,
                                                                           stop_after=nr_samples)
            logger.info(f"({time.time() - start:0.2f}) Epoch {epoch} on train dataset first {nr_samples}: "
                        f"MRR: {average_mrr:0.4f}; "
                        f"AvgRank: {average_rank:0.4f} "
                        f"Loss: {total_loss}")

        # training
        total_loss = 0
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
                    f'({time.time() - start:0.2f}) Training batch {i}/{len(train_loader)} loss: {train_batch_loss}')
            total_loss += train_batch_loss

        logger.info(f"({time.time() - start:0.2f}) Epoch {epoch} on training dataset: "
                    f"loss: {total_loss}")


if __name__ == "__main__":
    train_ce()
