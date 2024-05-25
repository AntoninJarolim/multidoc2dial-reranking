import logging
import time

import torch
import torch.nn.init as init
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM

from md2d_dataset import MD2DDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_mrr(preds, labels):
    """
    Compute Mean Reciprocal Rank (MRR).

    Arguments:
    pred -- Tensor of predicted scores.
    labels -- Tensor of true labels.

    Returns:
    MRR -- Mean Reciprocal Rank for the batch.
    """
    # Ensure labels are in the correct shape
    assert labels[0] == 1 and all(
        label == 0 for label in labels[1:]), "First label must be target and others non-targets"

    # Get the indices that would sort the predictions in descending order
    sorted_indices = torch.argsort(preds, descending=True)

    # Find the rank of the true target (which is always at index 0 in labels)
    rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1

    # Calculate the reciprocal rank
    return 1.0 / rank, rank


class CrossEncoder(torch.nn.Module):
    def __init__(self, bert_model, dropout_rate=0.1):
        super(CrossEncoder, self).__init__()

        self.bert_model = AutoModelForMaskedLM.from_pretrained(roberta_model)
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


roberta_model = "FacebookAI/xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(roberta_model)

train_dataset = MD2DDataset('data/DPR_pairs/DPR_pairs_train.jsonl',
                            tokenizer,
                            roberta_model,
                            shuffle=True)

test_dataset = MD2DDataset('data/DPR_pairs/DPR_pairs_test.jsonl',
                           tokenizer,
                           roberta_model)
val_dataset = MD2DDataset('data/DPR_pairs/DPR_pairs_validation.jsonl',
                          tokenizer,
                          roberta_model)

batch_size = 33
train_loader = DataLoader(train_dataset, batch_size=batch_size)

# only for evaluation - there are 10 negative samples and 1 positive sample
test_batch_size = 11
test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
val_loader = DataLoader(val_dataset, batch_size=test_batch_size)

cross_encoder = CrossEncoder(roberta_model)


def load_model(cross_encoder):
    load_path = "train_ce_e5step_dropout.pt"
    cross_encoder.load_state_dict(torch.load(load_path))
    logger.info(f"Model loaded successfully from {load_path}")


# load_model(cross_encoder)

cross_encoder.to(device)

optimizer = AdamW(cross_encoder.parameters(), lr=1e-5, weight_decay=1e-2)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10))

start = time.time()

num_epochs = 100
epoch = 0
try:
    for epoch in range(num_epochs):
        logger.info(f"({time.time() - start:0.2f}) Epoch: {epoch}")
        total_loss = 0

        # training
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

        # TESTING - MRR
        cross_encoder.eval()
        test_pred = []
        rr_total = 0
        rank_total = 0
        loss_total = 0

        for i, batch in enumerate(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                pred = cross_encoder(input_ids=batch['in_ids'], attention_mask=batch['att_mask'])
                loss = loss_fn(pred, batch['label'])
                loss_total += loss.item()
                # Compute MRR for the current batch
                batch_mrr, rank = compute_mrr(pred, batch['label'])
                rr_total += batch_mrr
                rank_total += rank

        num_batches = len(test_loader)
        average_mrr = rr_total / num_batches if num_batches > 0 else 0
        average_rank = rank_total / num_batches if num_batches > 0 else 0
        logger.info(f"({time.time() - start:0.2f}) Epoch {epoch} on test dataset: "
                    f"MRR: {average_mrr:0.4f}; "
                    f"AvgRank: {average_rank:0.4f} "
                    f"Loss: {loss_total}")


except KeyboardInterrupt:
    logger.info(f"Early stopping in epoch {epoch} by user Ctrl+C interaction.")

save_path = "cross_encoder.pt"
torch.save(cross_encoder.state_dict(), save_path)
logger.info(f"Model saved to {save_path}")
