import json
import sys

import jsonlines
import numpy as np
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DPR_SAMPLING_FROM = 20
DPR_SAMPLING_TO = 60

MAX_PSG_ID = 4110
neg_psg_counts = pd.Series([0 for x in range(MAX_PSG_ID)])

PARSING_DPR_RESULTS = True


def get_negative_passages(dpr_res):
    assert not PARSING_DPR_RESULTS, "This function is only for parsing DPR results."
    return [dpr_res["hard_negative_ctxs"][0]] + dpr_res["negative_ctxs"]


def get_positive_passage(dpr_res):
    if PARSING_DPR_RESULTS:
        return dpr_res["positive_passage"]
    else:
        return dpr_res["positive_ctxs"][0]["text"]


def write_test_samples(dpr_res, writer):
    q = dpr_res["question"].replace("[SEP]", "")

    examples = []
    if PARSING_DPR_RESULTS:
        # Positives and negatives are in DPR_result
        # pos/neg decides is_target value
        for psg in dpr_res["DPR_result"]:
            examples.append({
                "x": f"{q}[SEP]{psg['text']}",
                "label": int(psg['is_target'])
            })
    else:
        # Always take the first positive passage (there are never more)
        examples.append({
            "x": f"{q}[SEP]{dpr_res['positive_ctxs'][0]['text']}",
            "label": 1
        })
        # Negatives are in negative_ctxs and hard_negative_ctxs
        for neg_psg in get_negative_passages(dpr_res):
            examples.append({
                "x": f"{q}[SEP]{neg_psg['text']}",
                "label": 0
            })
    writer.write(examples)


def write_train_samples(dpr_res, writer, take_only_one=False):
    q = dpr_res["question"].replace("[SEP]", "")
    pos_passage = get_positive_passage(dpr_res)
    writer.write({
        "x": f"{q}[SEP]{pos_passage}",
        "label": 1
    })

    if PARSING_DPR_RESULTS:
        sampled_examples = dpr_res["DPR_result"][DPR_SAMPLING_FROM:DPR_SAMPLING_TO]
    else:
        sampled_examples = get_negative_passages(dpr_res)

    if take_only_one:
        # select random member from those, whose count value is smallest
        selected_ids = list(set([x['passage_id'] for x in sampled_examples]))
        min_psg_counts = neg_psg_counts[selected_ids].min()
        selected_with_lowest_count = [x for x in sampled_examples if neg_psg_counts[x['passage_id']] == min_psg_counts]
        if len(selected_with_lowest_count) == 0:
            selected_with_lowest_count = sampled_examples

        rnd_neg_psg = np.random.choice(selected_with_lowest_count)
        neg_psg_counts[rnd_neg_psg['passage_id']] += 1

        write_one_negative(rnd_neg_psg, q, writer)

    else:
        # Write each example if take_one is set to False
        for neg_psg in sampled_examples:
            write_one_negative(neg_psg, q, writer)


def write_one_negative(neg_psg, q, writer):
    writer.write({
        "x": f"{q}[SEP]{neg_psg['text']}",
        "label": 0
        # , "passage_id": f"{neg_psg['passage_id']}"
    })


def create_train_pairs(split, take_only_one=False):
    train_split = split == "train"
    if PARSING_DPR_RESULTS:
        path = f'data/DPR/DPR_{split}.json'
    else:
        path = f'data/DPR_train_bm25/dpr.multidoc2dial_all.structure.{split}.json'
    with open(path, mode="r") as f:
        data = json.load(f)

    bm25 = "_bm25" if not PARSING_DPR_RESULTS else ""
    sampling_postfix = f"_{DPR_SAMPLING_FROM}-{DPR_SAMPLING_TO}" if PARSING_DPR_RESULTS else ""
    out_file = f'data/DPR_pairs/DPR_pairs{bm25}_{split}{sampling_postfix}.jsonl'

    sampling = f"(sampling: <{DPR_SAMPLING_FROM}-{DPR_SAMPLING_TO}>)" if PARSING_DPR_RESULTS else ""
    print(f"Outputting train data {sampling} to: {out_file}")

    with jsonlines.open(out_file, mode="w") as writer:
        for i, d in enumerate(data):
            if train_split:
                write_train_samples(d, writer, take_only_one)
            else:
                write_test_samples(d, writer)

            if i % 100 == 0:
                print(f"Processing {i}/{len(data)}.")
    print(f"Processing finished.")


if __name__ == "__main__":
    split = sys.argv[1]
    take_only_one = bool(int(sys.argv[2]))
    PARSING_DPR_RESULTS = bool(int(sys.argv[3]))
    print(f"Split: {split}")
    print(f"Take only one: {take_only_one}")
    print(f"Parsing DPR results: {PARSING_DPR_RESULTS}")

    if split == "train":
        create_train_pairs("train", take_only_one)

    elif split == "test":
        print("Processing testing data.")
        create_train_pairs("test")
        print("Processing validation data.")
        create_train_pairs("validation")
    else:
        print("Please specify if you wish to create test or train data by test or train parameter.")
