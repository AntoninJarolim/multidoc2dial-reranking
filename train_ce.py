import json
import logging
import time
from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, DebertaV2Model, BertModel
from transformers import DebertaV2ForSequenceClassification

from md2d_dataset import MD2DDataset
from utils import mrr_metric, transform_batch, pred_recall_metric, calc_physical_batch_size, load_model, save_model, \
    save_best_model

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

    def log(self, epoch, dataset_name, gs, scheduler=None):
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

        if scheduler is not None:
            tb_writer.add_scalar(f"{dataset_name}/LR", scheduler.get_lr()[0], gs)


class BestMetricTracker:
    def __init__(self):
        self.best_mrr = None
        self.best_mrr_epoch = None
        self.best_mrr_gs = None

        self.gs = None

        self.current_best = None  # True if the current epoch is the best wrt MRR
        self.nr_not_improved = 0  # Counter of consecutive epochs without improvement (for early stopping)

    def is_current_best(self):
        # First epoch is always the best
        res = True if self.current_best is None else self.current_best
        return res

    def step(self, evaluation: EvaluationMetrics, epoch, gs):
        if self.best_mrr is None:
            # Only initialization
            self.current_best = True  # First epoch is always the best
            self.update_metrics(evaluation, epoch, gs)
        else:
            # Update the best metric
            self.current_best = True if self.best_mrr < evaluation.average_mrr else False
            if self.current_best:
                self.update_metrics(evaluation, epoch, gs)
            self.update_not_improved()

    def update_not_improved(self):
        if self.current_best:
            self.nr_not_improved = 0  # Reset counter if the current epoch is the best
        else:
            self.nr_not_improved += 1

    def update_metrics(self, evaluation, epoch, gs):
        self.best_mrr = evaluation.average_mrr
        self.best_mrr_epoch = epoch
        self.best_mrr_gs = gs


class CrossEncoder(torch.nn.Module):
    def __init__(self, bert_model_name, dropout_rate=0.1):
        super(CrossEncoder, self).__init__()

        m = AutoModelForSequenceClassification.from_pretrained(bert_model_name)
        if isinstance(m, DebertaV2ForSequenceClassification):
            bert_model = m.deberta
            hidden_size = m.config.hidden_size
            self.pooler = m.pooler
        else:
            assert isinstance(m, BertForSequenceClassification)
            bert_model = m.bert
            hidden_size = m.config.hidden_size

        # Model initialization
        self.bert_model = bert_model
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, 1)

        self.classifier.weight = m.classifier.weight
        self.classifier.bias = m.classifier.bias

        self.save_attention_weights = False  # Whether to save attention weights during evaluation
        self.last_attention_weights = None
        self.acc_attention_weights = None

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        cls_features = self.get_cls_features(input_ids, attention_mask, token_type_ids)
        cls_features = self.dropout(cls_features)
        logits = self.classifier(cls_features)
        # Flatten [batch_size, 1] to [batch_size]
        return logits.flatten()

    def get_cls_features(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True)
        if isinstance(self.bert_model, DebertaV2Model):
            # Deberta model doesnt do pooling automatically
            encoder_layer = outputs[0]
            pooled_output = self.pooler(encoder_layer)

            # Save attention weights
            if self.save_attention_weights:
                self.last_attention_weights = outputs[2]

            return pooled_output
        else:
            assert isinstance(self.bert_model, BertModel)
            # Bert model returns a tuple with the last hidden state and the pooler output
            return outputs[1]

    @torch.no_grad()
    def evaluate_ce(self, data_loader, loss_fn, take_n=0, save_all_losses=False) -> EvaluationMetrics:
        self.eval()
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

        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluation"):
            batch = transform_batch(batch, take_n)
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = self.process_large_batch(batch, 32)
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
        self.train()
        return EvaluationMetrics(average_mrr, sum_recalls, loss_total, loss_negative, loss_positive, sm_loss_sum)

    @torch.no_grad()
    def process_large_batch(self, batch, max_sub_batch_size):
        self.acc_attention_weights = []
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
            self.accumulate_attention_weights()

        # Merge predictions back
        merged_preds = torch.cat(preds, dim=0)
        return merged_preds

    def accumulate_attention_weights(self):
        if self.save_attention_weights == False:
            return 
        cpu_copy_weights = [layer.detach().cpu() for layer in self.last_attention_weights]
        self.acc_attention_weights.append(cpu_copy_weights)
        del self.last_attention_weights
        self.last_attention_weights = None


class WarmupCosineAnnealingWarmRestarts(LRScheduler):
    def __init__(self, optimizer, steps_per_epoch, num_epochs, warmup_percent, nr_restarts,
                 t_mult=1, eta_min=0, last_epoch=-1):

        # Calculate number of warmup steps
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = int(warmup_percent * total_steps)

        # Calculate nr of steps per restart
        annealing_steps = total_steps - warmup_steps
        first_restart_t0 = annealing_steps // nr_restarts

        # Save initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # Warmup specific parameters
        self.warmup_steps = warmup_steps

        # Cosine annealing specific parameters
        self.T_0 = first_restart_t0
        self.T_mult = t_mult
        self.eta_min = eta_min
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.T_0, T_mult=t_mult, eta_min=eta_min, last_epoch=last_epoch
        )
        super(WarmupCosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_lr = [(base_lr - self.eta_min) * self.last_epoch / self.warmup_steps + self.eta_min for base_lr in
                         self.base_lrs]
            return warmup_lr
        else:
            return self.cosine_scheduler.get_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_steps:
            self.last_epoch += 1
            warmup_lr = self.get_lr()
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            self.cosine_scheduler.step(epoch)


@dataclass
class TrainDataArgs:
    load_model_path: object = None
    save_model_path: str = "cross_encoder.pt"
    bert_model_name: str = "naver/trecdl22-crossencoder-debertav3"
    train_data_path: str = 'data/DPR_pairs/DPR_pairs_train.jsonl'
    test_data_path: str = 'data/DPR_pairs/DPR_pairs_test.jsonl'
    dont_save_model: bool = False

    def __repr__(self):
        class_name = self.__class__.__name__
        args_json = json.dumps(asdict(self), indent=4)
        return f"{class_name}:\n{args_json}"


@dataclass
class TrainHyperparameters:
    num_epochs: int = 30
    lr: float = 1e-5
    weight_decay: float = 1e-1
    positive_weight: float = 2
    dropout_rate: float = 0.1
    optimizer: object = None
    loss_fn: object = None
    stop_time: object = None  # in seconds
    label_smoothing: int = 0
    gradient_clip: float = None
    batch_size: int = 128
    warmup_percent: float = 0.1
    nr_restarts: int = 1
    lr_min: float = 1e-7
    test_every: str = "epoch"  # "epoch" or nr_gradient_steps
    evaluate_before_training: bool = False
    evaluation_take_n: int = 200

    def __repr__(self):
        class_name = self.__class__.__name__
        args_json = json.dumps(asdict(self), indent=4)
        return f"{class_name}:\n{args_json}"


def train_ce(data_args: TrainDataArgs, train_hyperparameters: TrainHyperparameters):
    logger.info(data_args)
    logger.info(train_hyperparameters)

    dropout_rate = train_hyperparameters.dropout_rate
    label_smoothing = train_hyperparameters.label_smoothing

    # Initialize cross encoder
    cross_encoder = CrossEncoder(data_args.bert_model_name, dropout_rate=dropout_rate)
    if data_args.load_model_path is not None:
        load_model(cross_encoder, data_args.load_model_path)
    cross_encoder.to(device)

    # Initialize datasets
    train_dataset = MD2DDataset(data_args.train_data_path,
                                data_args.bert_model_name,
                                label_smoothing=label_smoothing,
                                shuffle=False)
    test_dataset = MD2DDataset(data_args.test_data_path,
                               data_args.bert_model_name)

    # Start training
    try:
        best = training_loop(cross_encoder,
                             train_dataset,
                             test_dataset,
                             data_args.save_model_path,
                             **asdict(train_hyperparameters))
    except KeyboardInterrupt:
        logger.info(f"Early stopping by user Ctrl+C interaction.")
        best = None

    if not data_args.dont_save_model:
        save_model(cross_encoder, data_args.save_model_path)
    return best


def training_loop(cross_encoder,
                  train_dataset,
                  test_dataset,
                  save_model_path,
                  num_epochs,
                  lr,
                  weight_decay,
                  positive_weight,
                  dropout_rate,
                  optimizer,
                  loss_fn,
                  stop_time,
                  label_smoothing,
                  gradient_clip,
                  batch_size,
                  warmup_percent,
                  nr_restarts,
                  lr_min,
                  test_every,
                  evaluate_before_training,
                  evaluation_take_n
                  ):
    # Train data loader
    batch_size, gradient_accumulation_steps = calc_physical_batch_size(batch_size)
    logger.info(f"batch_size: {batch_size}\ngradient_accumulation_steps: {gradient_accumulation_steps}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    logger.info(f"Train data len: {len(train_loader)}")

    # Test data loader
    test_batch_size = 1  # Each line of testing data is a single batch
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
    steps_per_epoch = len(train_loader) * batch_size
    logger.info(f"Test data len: {len(test_loader)}")

    DEBUG = False
    if DEBUG:
        from utils import LimitedDataLoader
        test_loader = LimitedDataLoader(test_loader, 20)
        train_loader = LimitedDataLoader(train_loader, 500)

    # Optimizer initialization
    optimizer = optimizer or AdamW(cross_encoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad()
    loss_fn = loss_fn or nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))

    # Scheduler initialization
    scheduler = WarmupCosineAnnealingWarmRestarts(
        optimizer, steps_per_epoch, num_epochs, warmup_percent, nr_restarts, eta_min=lr_min)

    # Track the best metric to restore model later
    best_metric_tracker = BestMetricTracker()
    if test_every == "epoch":
        test_every = len(train_loader)

    # Create a list of gradient steps at which to evaluate the model
    max_nr_evaluations = (num_epochs * len(train_loader)) // test_every
    logger.info(f"Max nr evaluations: {max_nr_evaluations}")
    test_grad_steps = torch.arange(max_nr_evaluations) * test_every
    if not evaluate_before_training:
        test_grad_steps = test_grad_steps[1:]

    logger.info(f"evaluating at gradient steps: {test_grad_steps}")
    total_loss, loss_positive, loss_negative, grad_step = 0, 0, 0, 0
    for epoch in range(num_epochs):
        logger.info(f"({time.time() - start_t:0.2f})------------- Training epoch {epoch} started -------------")

        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):

            if grad_step in test_grad_steps:
                evaluation = cross_encoder.evaluate_ce(test_loader, loss_fn, take_n=evaluation_take_n)
                evaluation.log(epoch, "test", grad_step, scheduler)
                best_metric_tracker.step(evaluation, epoch, grad_step)

            # Training step
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = cross_encoder(input_ids=batch['in_ids'], attention_mask=batch['att_mask'])
            loss = loss_fn(pred, batch['label'])
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(cross_encoder.parameters(), gradient_clip)

            # Accumulate gradients to address gpu memory limitations
            if (grad_step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Update logging variables
            total_loss += loss.item()
            loss_negative, loss_positive = separate_losses(batch, loss_fn, pred, loss_negative, loss_positive)
            grad_step += 1

        # logging train losses after each epoch
        logger.info(f"({time.time() - start_t:0.2f}) "
                    f"Epoch: {epoch}; gs: {grad_step}; on training dataset: "
                    f"loss: {loss_positive + loss_negative}; "
                    f"positive-loss: {loss_positive:0.2f}; "
                    f"negative-loss: {loss_negative:0.2f} ")

        tb_writer.add_scalar("train/TotalLoss", total_loss, grad_step)
        tb_writer.flush()
        total_loss, loss_positive, loss_negative = 0, 0, 0

        if best_metric_tracker.is_current_best():
            save_best_model(cross_encoder, save_model_path)

        if best_metric_tracker.nr_not_improved > 2:
            logger.info(f"Early stopping due to no improvement in the last 3 epochs.")
            break

        # stopping based on max time to train
        if stop_time is not None and time.time() - start_t > stop_time:
            break

        # Checkpointing
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': cross_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'tracker': best_metric_tracker,
            }, "checkpoint.pt")


    # val_dataset = MD2DDataset('data/DPR_pairs/DPR_pairs_validation.jsonl',
    #                           bert_model_name)
    # val_loader = DataLoader(val_dataset, batch_size=test_batch_size)

    tb_writer.close()
    return best_metric_tracker


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
    pass

    # model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # features = tokenizer(['How many people live in Berlin?', 'How many people live in Berlin?'], [
    #    'Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.',
    #    'New York City is famous for the Metropolitan Museum of Art.'], padding=True, truncation=True,
    #                     return_tensors="pt")
    # bert_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    # cross_encoder = CrossEncoder(bert_model_name)
    # model.eval()
    # with torch.no_grad():
    #     scores = model(**features).logits
    #     print(scores)

    # train_ce()
