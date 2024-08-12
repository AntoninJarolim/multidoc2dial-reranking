import json

import numpy as np
import streamlit as st
import torch
from annotated_text import annotated_text
from streamlit_chat import message
from transformers import AutoTokenizer

import utils
from interpretability import attention_rollout
from md2d_dataset import preprocess_examples
from train_ce import CrossEncoder

st.set_page_config(layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Layout config
chat, _, explaining = st.columns([6, 1, 6])
data = np.random.randn(10, 1)

# DATA
EXAMPLE_VALIDATION_DATA = "data/examples/200_dialogues_reranking.json"
model_name = "naver/trecdl22-crossencoder-debertav3"


@st.cache_resource
def init_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cross_encoder = CrossEncoder(model_name)
    cross_encoder.save_attention_weights = True
    cross_encoder.bert_model.config.output_attentions = True
    cross_encoder = utils.load_model(cross_encoder, "CE_lr0.00010485388296357131_bs16.pt")
    cross_encoder.to(device)
    cross_encoder.eval()
    return cross_encoder, tokenizer


cross_encoder, tokenizer = init_model()


@st.cache_data
def get_data():
    return json.load(open(EXAMPLE_VALIDATION_DATA))


def split_to_tokens(text):
    return tokenizer.batch_decode(tokenizer(text)["input_ids"])[1:][:-1]


def split_utterance_history(raw_rarank_data):
    return list(map(lambda x: {
        "x": x["x"],
        "label": x["label"],
        "utterance_history": x["x"].split("[SEP]")[0],
        "passage": x["x"].split("[SEP]")[1]
    }, raw_rarank_data))


def mean_attention_heads(attentions_heads):
    return torch.mean(attentions_heads, dim=2)


def annt_list_2_colours(annotation_list, base_colour, colours):
    if annotation_list is None:
        return None
    assert base_colour in ["blue", "red", "green"], f"Base colour {base_colour} not supported"

    # Normalize annotation list and convert to ints (0-255)
    if not isinstance(annotation_list, torch.Tensor):
        annotation_list = torch.Tensor(annotation_list)

    normalized_tensor_list = annotation_list / torch.max(annotation_list)
    if colours == "nonlinear":
        transf_list = -torch.log(normalized_tensor_list)
        normalized_tensor_list = 1 - (transf_list / torch.max(transf_list))

    colour_range = (normalized_tensor_list * 255).type(torch.int64)

    if not (0 <= torch.min(colour_range)):
        raise ValueError(f"min: Conversion of {torch.min(colour_range)} to colour range failed")
    assert torch.max(colour_range) < 256, f"max: Conversion of {torch.max(colour_range)} to colour range failed"

    if base_colour == "blue":
        def conv_fce(x):
            return f'#1111{x:02x}'
    elif base_colour == "green":
        def conv_fce(x):
            return f'#11{x:02x}11'
    elif base_colour == "red":
        def conv_fce(x):
            return f'#{x:02x}0000'
    else:
        raise ValueError(f"Base colour {base_colour} not supported")

    coloured_list = [conv_fce(x) for x in colour_range]
    return [x if x not in ["#000000", "#111100", "#11011"] else ["#00000000"]
            for x in coloured_list]


def show_annotated_psg(passage_tokens, idx, is_grounding=False, annotation_scores=None, base_colour="blue",
                       colour_type="linear", score=None, hide_attention_colours=False, gt_label_list=None):
    """
    :param gt_label_list: list of True/False, True - GT word, False - no GT
    :param passage_tokens: list of strings of same length as annotation_sc
    :param annotation_scores: list of numbers used to colour the passage_text
    :param idx: ID of the passage to show
    :param is_grounding: Whether the passage is the grounding passage - 'id' will be coloured blue
    """
    assert not isinstance(passage_tokens, str)

    with st.container(border=True):
        col_idx, passage_col = st.columns([77, 1000])
        with col_idx:
            # Print psg index number in colour if grounding
            f'### :blue[{idx}]' if is_grounding else f'### {idx}'

        # Create default list of colours for each token
        colours_annotation_list = ["#00000000"] * len(passage_tokens)
        if annotation_scores is not None and not hide_attention_colours:
            colours_annotation_list = annt_list_2_colours(annotation_scores, base_colour, colour_type)

        if gt_label_list is None:
            gt_label_list = [None] * len(passage_tokens)

        with passage_col:
            if score is not None:
                f"#### {score}"

            coloured_passage = []
            for bg_colour, token, gt_label in zip(colours_annotation_list, passage_tokens, gt_label_list):
                fg_colour = "#FFFFFF" if not gt_label else "#4455FF"
                token = token.replace('$', '\$')
                coloured_passage.append((token, "", bg_colour, fg_colour))

            annotated_text(coloured_passage)


def create_grounding_annt_list(passage, grounded_agent_utterance, label):
    if label == 0:
        return split_to_tokens(passage), None

    # Create annotation list
    annotation_list = []
    broken_passage = []
    for reference in grounded_agent_utterance["references"]:
        # strip space, because it is sometimes appended at the end and the space is not in
        # the passage, leading to not finding part of passage containing this reference span
        ref_span = reference["ref_span"].strip(" ")

        if passage == ref_span:
            annotation_list.append(True)
            broken_passage.append(ref_span)
            break

        before, after = passage.split(ref_span)

        # Found reference is not at the beginning of the passage
        if before != "":
            annotation_list.append(False)
            broken_passage.append(before)  # Do not label this part of passage as GT

        annotation_list.append(True)
        broken_passage.append(ref_span)

        passage = after
        if passage == "":
            break

    # Append the remaining part of passage (if any)
    if passage != "":
        annotation_list.append(False)
        broken_passage.append(passage)

    unpack_passages = []
    unpack_annt_list = []
    for psg, annt in zip(broken_passage, annotation_list):
        psg_tokens = split_to_tokens(psg)
        for t in psg_tokens:
            unpack_passages.append(t)
            unpack_annt_list.append(annt)

    return unpack_passages, unpack_annt_list


data_dialogues = get_data()

# This variables will be set in configuration sidebar
set_data = {
    "current_dialogue": 0,
    "gt_label_colour": "#2222DD",
    "show_explanations": False,
    "show_relevance_score": False,
    "hide_attention_colours": False,
}

# CONFIGURATION SIDEBAR
with st.sidebar:
    "## Configuration"
    "### Dialog loading"
    dialogue_index = st.selectbox('Example dialog id:', list(range(len(data_dialogues))))
    set_data["current_dialogue"] = dialogue_index

    "### Togglers"
    set_data["show_explanations"] = st.toggle("Show explanations", set_data["show_explanations"])
    set_data["show_relevance_score"] = st.toggle("Show relevance score", set_data["show_relevance_score"])
    set_data["hide_attention_colours"] = st.toggle("Hide attention colours", set_data["hide_attention_colours"])

    selected_dialog = data_dialogues[set_data["current_dialogue"]]
    diag_turns = selected_dialog["dialog"]["turns"]
    rerank_dialog_examples = split_utterance_history(selected_dialog["to_rerank"])
    utterance_history, passage = [(d["utterance_history"], d["passage"])
                                  for d in rerank_dialog_examples
                                  if d["label"] == 1][0]
    last_user_utterance = utterance_history.split("agent: ")[0]
    last_user_utterance_id = [t['turn_id'] for t in diag_turns if t["utterance"] == last_user_utterance][0]
    grounded_agent_utterance_id = last_user_utterance_id  # Ids stats from 1 and agent utterance is the next after user
    grounded_agent_utterance = diag_turns[grounded_agent_utterance_id]
    nr_show_utterances = grounded_agent_utterance_id + 1

    # MID SECTION CHAT
    with chat:
        for utterance in diag_turns[:nr_show_utterances]:
            is_user = True if utterance["role"] == "user" else False
            message(f"{utterance['utterance']}", is_user=is_user)

        st.chat_input("Say something")


    # Cross encoder inference on current dialogue
    @st.cache_data
    def cross_encoder_inference(rerank_dialog_examples):
        max_to_rerank = 32
        pre_examples = preprocess_examples(rerank_dialog_examples, tokenizer, 512)
        batch = utils.transform_batch(pre_examples, max_to_rerank, device=device)
        preds = cross_encoder.process_large_batch(batch, max_to_rerank)
        att_weights = cross_encoder.get_attention_weights()

        # Sorting wrt predictions
        sorted_indexes = torch.argsort(preds, descending=True).cpu()
        reranked_examples = [rerank_dialog_examples[i] for i in sorted_indexes]
        sorted_preds = [preds[i] for i in sorted_indexes]

        # Compute attention rollout
        print(att_weights.shape)
        att_weights_norm_heads = mean_attention_heads(att_weights)
        print(att_weights_norm_heads.shape)
        batched_rollouts = attention_rollout(att_weights_norm_heads)
        print(batched_rollouts.shape)
        batched_rollouts_cls = batched_rollouts[:, 0, :]
        print(batched_rollouts[:, 0, :].shape)
        print(batched_rollouts[:, 0, :])
        print(batched_rollouts[:, 1, :])
        print(batched_rollouts[:, 2, :])
        print(batched_rollouts[:, 12, :])
        # batched_rollouts_cls = batched_rollouts[:, :, 0]

        # Reranking rollouts on prediction scores
        reranked_att_weights = att_weights[:, sorted_indexes, :, 0, :]
        reranked_rollouts = [batched_rollouts_cls[i] for i in sorted_indexes]
        return {
            "reranked_examples": reranked_examples,
            "sorted_predictions": sorted_preds,
            "reranked_rollouts": reranked_rollouts,
            "attention_weights": att_weights_norm_heads,
            "att_weights_cls": reranked_att_weights  # (L, B, H, S)
        }


    # RIGHT SECTION EXPLAINING features
    with (explaining):
        "### Reranked results and attention visualizations"
        gt_tab, att_rollout_tab, raw_att_tab = st.tabs(["Ground Truth", "Attention Rollout", "Raw Attention"])

        with gt_tab:
            with st.container(height=800):
                for i, example in enumerate(rerank_dialog_examples, start=1):
                    passage_list, gt_labels = create_grounding_annt_list(example["passage"],
                                                                         grounded_agent_utterance,
                                                                         example["label"])
                    show_annotated_psg(passage_list, i, example["label"], gt_label_list=gt_labels)

        inf_out = cross_encoder_inference(rerank_dialog_examples)


        def visualize_example(scores_to_colour, colour_type="linear"):

            for idx, (r_example, score_colour, score) in enumerate(
                    zip(inf_out['reranked_examples'], scores_to_colour, inf_out['sorted_predictions']),
                    start=1):
                score = score if set_data["show_relevance_score"] else None
                passage_list, gt_labels = create_grounding_annt_list(r_example['x'],
                                                                     grounded_agent_utterance,
                                                                     r_example["label"])
                score_colour = score_colour[1:][:-1][:len(passage_list)]
                show_annotated_psg(passage_list, idx, is_grounding=r_example["label"], annotation_scores=score_colour,
                                   base_colour="green", colour_type=colour_type, score=score,
                                   hide_attention_colours=set_data["hide_attention_colours"],
                                   gt_label_list=gt_labels)


        with att_rollout_tab:

            linear_tab, not_linear_tab = st.tabs(["Linear colour", "Non-linear colours"])
            with linear_tab:
                with st.container(height=800):
                    visualize_example(inf_out['reranked_rollouts'], colour_type="linear")

            with not_linear_tab:
                with st.container(height=800):
                    visualize_example(inf_out['reranked_rollouts'], colour_type="nonlinear")

        with raw_att_tab:
            l_idx = st.number_input("Raw Attention Layer", min_value=1, max_value=24)
            att_weights = inf_out["att_weights_cls"]  # (L, B, H, S)

            head_tabs = st.tabs([f"Head {i + 1}" for i in range(att_weights.shape[2])])
            for h_idx, head_tab in enumerate(head_tabs):
                with head_tab:
                    att_map = att_weights[l_idx, :, h_idx, :]
                    linear_tab, not_linear_tab = st.tabs(["Linear colour", "Non-linear colours"])
                    with linear_tab:
                        with st.container(height=800):
                            visualize_example(att_map, colour_type="linear")

                    with not_linear_tab:
                        with st.container(height=800):
                            visualize_example(att_map, colour_type="nonlinear")
