import pandas as pd
import torch

from custom_data_utils.utils import create_grounding_annt_list, split_to_tokens
from visualization_data import InferenceDataProvider, init_model


def get_scores_passage_only(inf_out_all_passages, passage_id, sep_index, nr_tokens_trunc_passage):
    def get_one_score(score_list):
        return score_list[1:][:-1][sep_index + 1:][:nr_tokens_trunc_passage]

    scores = {}
    for label in score_labels:
        scores[label] = get_one_score(inf_out_all_passages[label][passage_id])

    return scores


def get_inf_data():
    df_data = []
    for diag_id in annotated_ids:

        inf_out = data_provider.get_dialog_inference_out(diag_id, nr_annotated_passages)
        max_score_len = inf_out["reranked_rollouts"][0].size(0) - 2  # -2 for [CLS] and [EOS]

        for example_id, example in enumerate(inf_out["reranked_examples"]):
            if "gpt_references" in example:
                gt_label = 1  # create_grounding_annt_list expects gt_label 1, otherwise it will not return any refs
                passage_tokens, gpt_labels_refs, failed_refs = create_grounding_annt_list(example["passage"],
                                                                                          example["gpt_references"],
                                                                                          gt_label,
                                                                                          tokenizer,
                                                                                          return_failed=True)
                if not failed_refs:
                    tokens = split_to_tokens(example["x"], tokenizer)[:max_score_len]

                    sep_index = [i for i, x in enumerate(tokens) if x == "[SEP]"][0]
                    nr_tokens_trunc_passage = len(tokens[sep_index + 1:])

                    record = {
                        "passage": example["passage"],
                        "gpt-labels-refs": gpt_labels_refs[:nr_tokens_trunc_passage],
                        "passage-tokens": passage_tokens[:nr_tokens_trunc_passage],
                        "diag_sep_passage": example["x"],
                    }

                    passage_scoring = get_scores_passage_only(inf_out, example_id, sep_index, nr_tokens_trunc_passage)
                    record.update(passage_scoring)

                    for label in score_labels:
                        if not (len(record["gpt-labels-refs"]) == len(record[label])):
                            raise AssertionError(
                                f"Length mismatch: {len(record['gpt-labels-refs'])} vs {len(record['grad-sam-refs'])}")

                    df_data.append(record)
    return df_data


if __name__ == "__main__":
    cross_encoder, tokenizer = init_model()

    data_provider = InferenceDataProvider(cross_encoder, tokenizer)
    nr_annotated_passages = 16
    nr_annotated_dialogues = 103
    annotated_ids = data_provider.get_valid_dialog_ids()[:nr_annotated_dialogues]
    score_labels = [
        'reranked_rollouts',
        'grad_sam_scores',
        'att_cat_scores',
    ]

    df = pd.DataFrame(get_inf_data())

    def conv_boolean(x):
        return 1 if x else -1

    df['gpt-labels-refs'] = df['gpt-labels-refs'].apply(lambda x: torch.Tensor([float(conv_boolean(i)) for i in x]))

    df.to_pickle("data/token_scores.pkl")

    print(df)
