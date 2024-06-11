import json
import sys

import jsonlines
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DPR_SAMPLING_FROM = 50
DPR_SAMPLING_TO = 60


def write_test_samples(dpr_res, writer):
    q = dpr_res["question"].replace("[SEP]", "")

    examples = []
    for psg in dpr_res["DPR_result"]:
        examples.append({
            "x": f"{q}[SEP]{psg['text']}",
            "label": int(psg['is_target'])
        })
    writer.write(examples)


def write_train_samples(dpr_res, writer):
    q = dpr_res["question"].replace("[SEP]", "")
    pos_passage = dpr_res["positive_passage"]
    writer.write({
        "x": f"{q}[SEP]{pos_passage}",
        "label": 1
    })
    for neg_psg in dpr_res["DPR_result"][DPR_SAMPLING_FROM:DPR_SAMPLING_TO]:
        writer.write({
            "x": f"{q}[SEP]{neg_psg['text']}",
            "label": 0
        })


def create_train_pairs(split):
    train_split = split == "train"
    with open(f'data/DPR/DPR_{split}.json', mode="r") as f:
        data = json.load(f)

    out_file = f'data/DPR_pairs/DPR_pairs_{split}_{DPR_SAMPLING_FROM}-{DPR_SAMPLING_TO}.json'
    print(f"Outputting train data (sampling: <{DPR_SAMPLING_FROM}-{DPR_SAMPLING_TO}>) to: "
          f"{out_file}")
    with jsonlines.open(out_file, mode="w") as writer:
        for i, d in enumerate(data):
            if train_split:
                write_train_samples(d, writer)
            else:
                write_test_samples(d, writer)

            if i % 100 == 0:
                print(f"Processing {i}/{len(data)}.")
    print(f"Processing finished.")


if __name__ == "__main__":
    split = sys.argv[1]

    if split == "--train":
        create_train_pairs("train")

    elif split == "--test":
        print("Processing testing data.")
        create_train_pairs("test")
        print("Processing validation data.")
        create_train_pairs("validation")
    else:
        print("Please specify if you wish to create test or train data by --test or --train parameter.")
