import json
import logging
import os
import pickle
import random as rnd

import jsonlines
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def break_to_pair(sentence):
    res = sentence.split("[SEP]")
    return res[0], res[1]


class MD2DDataset(IterableDataset):
    def __init__(self, data_path, tokenizer_name, shuffle=False, retokenize=False):
        super(MD2DDataset).__init__()

        # tokenizer
        self.tokenizer_name = tokenizer_name.replace("/", "_")

        self.preprocessed_f_handle = None
        self.preprocessed_f = f"data/{self.tokenizer_name}/{data_path.split('/')[-1]}"
        if not os.path.exists(self.preprocessed_f) or retokenize:
            os.makedirs(os.path.dirname(self.preprocessed_f), exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.preprocess_file(tokenizer, data_path)

        # data loading/iterating management
        self.shuffle = shuffle
        self._line_offsets = self.index_dataset()
        self.offset = 0

    def preprocess_file(self, tokenizer, data_path):
        logger.info(f"Preprocessing data with {self.tokenizer_name} tokenizer.")
        with open(data_path, 'r') as infile, jsonlines.open(self.preprocessed_f, 'w') as outfile:
            for i, line in enumerate(infile):
                obj = json.loads(line.strip())
                pair = break_to_pair(obj['x'])
                tokenized = tokenizer([pair], return_tensors="pt", padding="max_length", truncation=True,
                                      max_length=128)

                obj['in_ids'] = tokenized["input_ids"][0].tolist()
                obj['att_mask'] = tokenized["attention_mask"][0].tolist()
                # print(obj['in_ids'])
                # print(obj['att_mask'])
                # print(tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0]))
                del obj['x']
                outfile.write(obj)
                if i % 1000 == 0:
                    print(f"{i}")

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
        return json.loads(self.preprocessed_f_handle.readline().strip())

    def __len__(self):
        return len(self._line_offsets)

    def __next__(self):
        if self.offset >= len(self.order):
            if not self.preprocessed_f_handle.closed:
                self.preprocessed_f_handle.close()
            raise StopIteration
        example = self.get_example(self.order[self.offset])
        self.offset += 1

        example['in_ids'] = torch.tensor(example['in_ids'])
        example['att_mask'] = torch.tensor(example['att_mask'])
        example['label'] = torch.tensor(example['label']).float()
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
                                tokenizer,
                                roberta_model,
                                shuffle=True)

    batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    # todo: mby add shuffle = True

    for i, batch in enumerate(train_loader):
        print(i)
        print(batch)
        print(batch.keys())
        pass
        break
