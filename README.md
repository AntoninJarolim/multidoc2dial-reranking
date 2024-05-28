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