## Description

Reranking retrieved passages from DPR using cross-encoder.

## Data

MultiDoc2Dial repository has script `scripts/run_data_preprocessing_dpr.sh` which prepares data for DPR.

## DPR

Script `DPR.py` uses pretrained DPR to find relevant passages.

## Preprocessing

Preprocessing for Cross-Encoder is performed in `preprocessing.py` script. It takes output from previous step and
creates
examples in the following jsonlines format.

```jsonlines
{ "x": "q[SEP]p+", "label": 1}
{ "x": "q[SEP]p-", "label": 0}
...
```

