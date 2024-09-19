import pandas as pd
from sklearn.metrics import f1_score, recall_score
import torch
from custom_data_utils import utils
from IPython.display import display, HTML

def get_df():
    # Read the data
    df = pd.read_pickle("../data/token_scores.pkl")
    df["gpt_any_refs"] = df["gpt_labels_refs_bool"].apply(lambda x: any(x))
    df["gpt_any_refs"].value_counts()

    # Clean the data
    df.drop(df[df["gpt_any_refs"] == False].index, inplace=True)
    df = df.reset_index(drop=True)
    return df


def calc_f1_score(x, y):
    if x is None or y is None:
        return None
    return f1_score(x, y)


def get_top_k_indexes(scores: torch.Tensor, top_k=5):
    top_k_scores, top_k_indexes = torch.topk(scores.flatten(), top_k)
    conv_ids = torch.stack(torch.unravel_index(top_k_indexes, scores.shape)).T
    # Ensure only top_k, because torch.topk returns more than k, when duplicated values
    return conv_ids[:top_k]


def threshold_score(scores, top_k):
    top_k = top_k if top_k < len(scores) else len(scores)
    top_k_indexes = get_top_k_indexes(scores, top_k)
    thresholded_scores = torch.ones_like(scores) * -1  # Negative label in data is -1
    thresholded_scores[top_k_indexes] = 1
    return thresholded_scores


def show_highlighted_passage(record, gt_bools_label, scores_label):
    highlighted_passage = utils.create_highlighted_passage(record["passage_tokens"],
                                                           gt_bools_label,
                                                           scores_label,
                                                           'green',
                                                           'linear')
    display(HTML("\n".join(highlighted_passage)))


def sanity_check(score_label, threshold, gt_label, running_f1, gt_record_index=22):
    print(f"Label: {score_label} with threshold {threshold:.2f} has f1 mean: {running_f1.mean():0.4f}")

    # Print f1 for random sample
    rnd_record_id = inds_with_gt_label[gt_record_index]
    record = df.iloc[rnd_record_id]
    print(f"rnd-record({rnd_record_id}) f1: {running_f1.iloc[rnd_record_id]:0.4f}")

    # Show threshold annotation
    gt_label_bool = gt_label + "_bool"
    record["running_threshold"][record["running_threshold"] == -1] = 0  # -1 to calc f1, 0 for vizualizations
    show_highlighted_passage(record, record[gt_label_bool], record["running_threshold"])

def compute_running_thresholds(df_apply, score_label, top_k_tokens, gt_label):
    df_apply["running_threshold"] = df_apply[score_label].apply(lambda x: threshold_score(x, top_k_tokens))
    running_f1 = df_apply.apply(lambda x: calc_f1_score(x["running_threshold"], x[gt_label]), axis=1)
    return running_f1

df = get_df()
df["has_gt_labels"] = df["gt_labels_refs_bool"].apply(lambda x: x is not None and any(x))
inds_with_gt_label = df[df["has_gt_labels"]].index.to_list()
