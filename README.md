## Description

Reranking retrieved passages from DPR using cross-encoder.

## Data

MultiDoc2Dial repository has script `scripts/run_data_preprocessing_dpr.sh` which prepares data for DPR.

## DPR retrieval

Script `DPR.py` uses pretrained DPR to find relevant passages. It looks for passages
in  `../multidoc2dial/data/mdd_dpr/dpr.psg.multidoc2dial_all.structure.json` folder
and `../multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.{split}.json` folder for queries.

## Preprocessing

### Traininig

Preprocessing DPR retrieval results for Cross-Encoder training is performed in `preprocessing.py` script. Cross encoder
is trained using 0 or 1 labels for negative or positive passage, for negative samples it takes DPR retrieval results on
ranks 10-20 and for positive example, it adds true positive passage with label 1. Each example is on separate line,
therefore, shuffling dataset for mini-batches is convenient.  
Following command creates examples in the following jsonlines format.

`python preprocessing.py --train`

```jsonlines
{ "x": "q[SEP]p+", "label": 1}
{ "x": "q[SEP]p-", "label": 0}
...
```

### Validation

R@k metric is used for validation, therefore the true positive passage is not added directly. All retrieval results are
in one line.

`python preprocessing.py --test`

```jsonlines
[ {"x": "q1[SEP]p-", "label": 0}, {"x": "q1[SEP]p+", "label": 1}, {"x": "q1[SEP]p-", "label": 0}]
[ {"x": "q2[SEP]p-", "label": 0}, {"x": "q2[SEP]p-", "label": 0}, {"x": "q2[SEP]p+", "label": 1}]
...
```

## Data loading

Data loader class deriving from `IterableDataset` is implemented in `md2d_dataset.py`. It uses line index to access
examples. As argument, it expects `jsonl` file from previous step and tokenizer which should be used to tokenize the
data. For the data not to be tokenized every time during training, it tokenizes the data to the
`f"data/{self.tokenizer_name}/{split_json}"` folder. Then, it creates line index to the same folder and prepends files
with `locache.pkl` .

## Others

### Baseline recall

`python main.py --compute_recall_at_k` computes R@1, R@5, R@10, R@50 and R@200.
Computing recall for data/DPR_pairs/DPR_pairs_test.jsonl.

```
R@1: 37.88
R@5: 67.22
R@10: 76.94
R@50: 90.69
R@200: 96.75
```

Computing recall for data/DPR_pairs/DPR_pairs_validation.jsonl.

```
R@1: 36.82
R@5: 66.63
R@10: 75.34
R@50: 89.60
R@200: 95.50    
```
