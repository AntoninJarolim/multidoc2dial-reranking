import json
import logging
import os
import pickle
import random as rnd

import jsonlines
import torch
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def break_to_pair(sentence):
    res = sentence.split("[SEP]")
    return res[0], res[1]


def preprocess_example(obj, tokenizer, model_max_length):
    pair = break_to_pair(obj['x'])
    tokenized = tokenizer(pair[0], pair[1],
                          return_tensors="pt", padding="max_length",
                          truncation=True, max_length=model_max_length)
    obj['in_ids'] = tokenized["input_ids"][0].tolist()
    obj['att_mask'] = tokenized["attention_mask"][0].tolist()
    obj['tt_ids'] = tokenized["token_type_ids"][0].tolist()
    del obj['x']
    return obj


class MD2DDataset(IterableDataset):
    def __init__(self, data_path, tokenizer_name, shuffle=False, retokenize=False, label_smoothing=0):
        super(MD2DDataset).__init__()

        # tokenizer
        self.tokenizer_name = tokenizer_name.replace("/", "_")

        self.preprocessed_f_handle = None

        # computes tokenized json files from path e.g.'data/DPR_pairs/DPR_pairs_test.jsonl'
        # check if data_path json file was tokenized before
        self.preprocessed_f = f"data/{self.tokenizer_name}/{data_path.split('/')[-1]}"
        if not os.path.exists(self.preprocessed_f) or retokenize:
            os.makedirs(os.path.dirname(self.preprocessed_f), exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            model = AutoModelForMaskedLM.from_pretrained(tokenizer_name)
            self.model_max_length = model.config.max_position_embeddings
            self.preprocess_file(tokenizer, data_path)

        # data loading/iterating management
        self.label_smoothing = label_smoothing
        self.shuffle = shuffle
        self._line_offsets = self.index_dataset()
        self.offset = 0

    def preprocess_file(self, tokenizer, data_path):
        with open(data_path, 'r') as infile:
            total_lines = sum(1 for _ in infile)

        logger.info(f"Preprocessing data with {self.tokenizer_name} tokenizer.")
        with open(data_path, 'r') as infile, jsonlines.open(self.preprocessed_f, 'w') as outfile:
            for i, line in tqdm(enumerate(infile), total=total_lines, desc="Preprocessing jsonl file."):
                obj = json.loads(line.strip())
                if type(obj) is list:
                    tokenized = []
                    for o in obj:
                        tokenized.append(preprocess_example(o, tokenizer, self.model_max_length))
                    outfile.write(tokenized)
                else:
                    preprocess_example(obj, tokenizer, self.model_max_length)
                    outfile.write(obj)

    def index_dataset(self):
        """
        Makes index of dataset. Which means that it finds offsets of the samples lines.
        Author: Martin Docekal, modified by Martin Fajcik
        """

        lo_cache = self.preprocessed_f + "locache.pkl"
        if os.path.exists(lo_cache):
            logger.info(f"Using cached line offsets from {lo_cache}")
            with open(lo_cache, "rb") as f:
                return pickle.load(f)
        else:
            logger.info(f"Getting lines offsets in {self.preprocessed_f}")
            return self._index_dataset(lo_cache)

    def _index_dataset(self, lo_cache):
        line_offsets = [0]
        with open(self.preprocessed_f, "rb") as f:
            while f.readline():
                line_offsets.append(f.tell())
        del line_offsets[-1]
        # cache file index
        with open(lo_cache, "wb") as f:
            pickle.dump(line_offsets, f)
        return line_offsets

    def get_example(self, n: int):
        """
        Get n-th line from dataset file.
        :param n: Number of line you want to read.
        :type n: int
        :return: the line
        :rtype: str
        Author: Martin Docekal, modified by Martin Fajcik
        """
        if self.preprocessed_f_handle.closed:
            self.preprocessed_f_handle = open(self.preprocessed_f)

        self.preprocessed_f_handle.seek(self._line_offsets[n])
        example = json.loads(self.preprocessed_f_handle.readline().strip())
        return example

    def __len__(self):
        return len(self._line_offsets)

    def __next__(self):
        if self.offset >= len(self.order):
            if not self.preprocessed_f_handle.closed:
                self.preprocessed_f_handle.close()
            raise StopIteration
        example = self.get_example(self.order[self.offset])
        self.offset += 1

        if type(example) is not list:
            example = self.transform_example(example)
        else:
            example2 = []
            for o in example:
                example2.append(self.transform_example(o))
            example = example2

        return example

    def transform_example(self, example):
        example['in_ids'] = torch.tensor(example['in_ids'])
        example['att_mask'] = torch.tensor(example['att_mask'])
        example['tt_ids'] = torch.tensor(example['tt_ids'])

        # Label smoothing for two classes
        if self.label_smoothing != 0:
            eps = self.label_smoothing
            label = torch.tensor(example['label']).float() * (1 - eps) + eps / 2
        else:
            label = torch.tensor(example['label']).float()
        example['label'] = label
        return example

    def __iter__(self):
        self.preprocessed_f_handle = open(self.preprocessed_f)
        self.order = list(range(len(self)))
        if self.shuffle:
            logger.info("Shuffling file index...")
            rnd.shuffle(self.order)
        self.offset = 0
        return self


if __name__ == "__main__":
    roberta_model = "FacebookAI/xlm-roberta-base"

    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

    train_dataset = MD2DDataset('data/DPR_pairs/DPR_pairs_train.jsonl',
                                roberta_model,
                                shuffle=True)

    batch_size = 11
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    # todo: mby add shuffle = True

    test_dataset = MD2DDataset('data/DPR_pairs/DPR_pairs_test.jsonl',
                               roberta_model)
    test_loader = DataLoader(test_dataset, batch_size=1)

    for i, batch in enumerate(test_loader):
        print(i)
        print(batch)
        pass
        break
