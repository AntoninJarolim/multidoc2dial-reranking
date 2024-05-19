import json

import jsonlines
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_pairs(split):
    with open(f'data/DPR/DPR_{split}.json', mode="r") as f:
        data = json.load(f)

    with jsonlines.open(f'data/DPR_pairs/DPR_pairs_{split}.jsonl', mode="w") as writer:
        for i, d in enumerate(data):
            q = d["question"].replace("[SEP]", "")
            pos_passage = d["positive_passage"]

            writer.write({
                "x": f"{q}[SEP]{pos_passage}",
                "label": 1
            })

            for neg_psg in d["DPR_result"][10:]:
                writer.write({
                    "x": f"{q}[SEP]{neg_psg['text']}",
                    "label": 0
                })

            if i % 100 == 0:
                print(f"Processing {i}/{len(data)}.")


if __name__ == "__main__":
    for split in ["train", "validation", "test"]:
        create_pairs(split)
