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


def try_get_gt_labels(example, gt_label, grounded_agent_utterance):
    gt_labels_refs = None
    if example["label"]:
        _, gt_labels_refs = create_grounding_annt_list(example["passage"],
                                                       grounded_agent_utterance["references"],
                                                       gt_label,
                                                       tokenizer)
    return gt_labels_refs


def get_inf_data():
    df_data = []
    for diag_id in annotated_ids:

        # Needed to calculate vector of booleans saying which one is relevant
        _, grounded_agent_utterance, _, _ \
            = data_provider.get_dialog_out(diag_id)

        inf_out = data_provider.get_dialog_inference_out(diag_id, nr_annotated_passages)
        max_score_len = inf_out["reranked_rollouts"][0].size(0) - 2  # -2 for [CLS] and [EOS]

        for example_id, example in enumerate(inf_out["reranked_examples"]):
            if "gpt_references" in example and example["gpt_references"] != []:
                gt_label = 1  # create_grounding_annt_list expects gt_label 1, otherwise it will not return any refs
                passage_tokens, gpt_labels_refs, failed_refs = create_grounding_annt_list(example["passage"],
                                                                                          example["gpt_references"],
                                                                                          gt_label,
                                                                                          tokenizer,
                                                                                          return_failed=True)
                gt_labels_refs = try_get_gt_labels(example, gt_label, grounded_agent_utterance)

                if not failed_refs:
                    tokens = split_to_tokens(example["x"], tokenizer)[:max_score_len]

                    sep_index = [i for i, x in enumerate(tokens) if x == "[SEP]"][0]
                    nr_tokens_trunc_passage = len(tokens[sep_index + 1:])

                    record = {
                        "passage": example["passage"],
                        "gpt_labels_refs_bool": gpt_labels_refs[:nr_tokens_trunc_passage],
                        "passage_tokens": passage_tokens[:nr_tokens_trunc_passage],
                        "diag_sep_passage": example["x"],
                        "gpt_refs": example["gpt_references"],
                        "gt_labels_refs_bool": gt_labels_refs[:nr_tokens_trunc_passage]
                    }

                    passage_scoring = get_scores_passage_only(inf_out, example_id, sep_index, nr_tokens_trunc_passage)
                    record.update(passage_scoring)

                    for label in score_labels:
                        if not (len(record["gpt_labels_refs"]) == len(record[label])):
                            raise AssertionError(
                                f"Length mismatch: {len(record['gpt_labels_refs'])} vs {len(record['grad_sam_refs'])}")

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


    def conv_boolean_arr(x):
        return torch.Tensor([float(1 if i else -1) for i in x])


    df['gpt_labels_refs'] = df['gpt_labels_refs_bool'].apply(lambda x: conv_boolean_arr(x))
    df['gpt_labels_refs'] = df['gt_labels_refs_bool'].apply(lambda x: conv_boolean_arr(x))

    df.to_pickle("data/token_scores.pkl")

    print(df)
