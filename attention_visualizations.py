import json

import numpy as np
import streamlit as st
import torch
from annotated_text import annotated_text
from streamlit_chat import message
from transformers import AutoTokenizer

import utils
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


@st.cache_data
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


def attention_rollout(attentions, r):
    """Computes attention rollout from the given list of attention matrices.
    https://arxiv.org/abs/2005.00928
    """
    # Get matrix from each layer of corresponding batch
    As = []
    for l in range(len(attentions)):
        As.append(attentions[l][r])

    rollout = As[0]
    for A in As[1:]:
        rollout = torch.matmul(
            0.5 * A + 0.5 * torch.eye(A.shape[1], device=A.device),
            rollout
        )

    return rollout


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


# RIGHT SECTION EXPLAINING features
def annt_list_2_colours(annotation_list, base_colour, colours):
    assert base_colour in ["blue", "red", "green"], f"Base colour {base_colour} not supported"

    # Normalize annotation list and convert to ints (0-255)
    if not isinstance(annotation_list, torch.Tensor):
        annotation_list = torch.Tensor(annotation_list)

    normalized_tensor_list = annotation_list / torch.max(annotation_list)
    if colours == "nonlinear":
        transf_list = -torch.log(normalized_tensor_list)
        normalized_tensor_list = 1 - (transf_list / torch.max(transf_list))

    colour_range = (normalized_tensor_list * 255).type(torch.int64)

    assert 0 <= torch.min(colour_range), f"min: Conversion of {torch.min(colour_range)} to colour range failed"
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
    return [x if x not in ["#000000", "#111100", "#11011"] else None
            for x in coloured_list]


def show_annotated_psg(passage_text, idx,
                       is_grounding=False,
                       annotation_list=None,
                       base_colour="blue",
                       colours="linear",
                       score=None,
                       hide_attention_colours=False,
                       ):
    """
    :param passage_text: string if annotation_list is None, else list of strings of same length as annotation_list
    :param annotation_list: list of numbers used to colour the passage_text
    :param idx: ID of the passage to show
    :param is_grounding: Whether the passage is the grounding passage - id will be shown in colour
    """
    with st.container(border=True):
        col_idx, passage_col = st.columns([77, 1000])
        with col_idx:
            if is_grounding:
                st.subheader(f':blue[{idx}]')
            else:
                st.subheader(idx)

        with passage_col:
            if score is not None:
                f"#### {score.cpu()}"

            if annotation_list is not None:
                colours_annotation_list = annt_list_2_colours(annotation_list, base_colour, colours)
                coloured_passage = []
                for colour, text in zip(colours_annotation_list, passage_text):
                    if colour is None:
                        coloured_passage.append(text.replace('$', '\$'))
                    else:
                        text_tokens = split_to_tokens(text)
                        for token in text_tokens:
                            token = token.replace('$', '\$')
                            if hide_attention_colours:
                                coloured_passage.append(token + " ")
                            else:
                                coloured_passage.append((token, "", colour))

                annotated_text(coloured_passage)
            else:
                passage_text = passage_text.replace('$', '\$')
                passage_text


def create_grounding_annt_list(passage, grounded_agent_utterance, label):
    if label == 0:
        return passage, None

    # Create annotation list
    annotation_list = []
    broken_passage = []
    for reference in grounded_agent_utterance["references"]:
        ref_span = reference["ref_span"]

        if passage == ref_span:
            annotation_list.append(1)
            broken_passage.append(ref_span)
            break

        before, after = passage.split(ref_span)

        # Found reference is not at the beginning of the passage
        if before != "":
            annotation_list.append(0)
            broken_passage.append(before)  # Do not label this part of passage as GT

        annotation_list.append(1)
        broken_passage.append(ref_span)

        passage = after
        if passage == "":
            break

    # Append the remaining part of passage (if any)
    if passage != "":
        annotation_list.append(0)
        broken_passage.append(passage)
    return broken_passage, annotation_list


with (explaining):
    "### Reranked results and attention visualizations"
    gt_tab, att_rollout_tab, raw_att_tab = st.tabs(["Ground Truth", "Attention Rollout", "Raw Attention"])

    with gt_tab:
        with st.container(height=800):
            gt_label = "GT"

            for i, example in enumerate(rerank_dialog_examples, start=1):
                passage_list, annotation_list = create_grounding_annt_list(example["passage"],
                                                                           grounded_agent_utterance,
                                                                           example["label"])
                show_annotated_psg(passage_list, i, example["label"], annotation_list)

    with att_rollout_tab:
        try:
            max_to_rerank = 32
            pre_examples = preprocess_examples(rerank_dialog_examples, tokenizer, 512)
            batch = utils.transform_batch(pre_examples, max_to_rerank, device=device)
            pred = cross_encoder.process_large_batch(batch, max_to_rerank)

            # Mean across heads
            att_weights = [torch.mean(t, dim=1) for t in cross_encoder.get_attention_weights()]
            batched_rollout = []
            for r in range(max_to_rerank):
                rollout = attention_rollout(att_weights, r)
                CLS_token_rollout = rollout[0, :]
                batched_rollout.append(CLS_token_rollout)

            # Reranking based on prediction scores
            sorted_indexes = torch.argsort(pred, descending=True)
            reranked_examples = [rerank_dialog_examples[i] for i in sorted_indexes]
            reranked_rollouts = [batched_rollout[i] for i in sorted_indexes]
            preds = [pred[i] for i in sorted_indexes]


            def visualize_rollout(colours="linear"):
                for idx, (example, rollout, score) in enumerate(zip(reranked_examples, reranked_rollouts, preds),
                                                                start=1):
                    score = score if set_data["show_relevance_score"] else None
                    psg = split_to_tokens(example["passage"])
                    rollout = rollout[1:][:-1][:len(psg)]
                    show_annotated_psg(psg, idx, is_grounding=example["label"],
                                       annotation_list=rollout, base_colour="green",
                                       colours=colours, score=score,
                                       hide_attention_colours=set_data["hide_attention_colours"])


            linear_tab, not_linear_tab = st.tabs(["Linear colour", "Non-linear colours"])
            with linear_tab:
                with st.container(height=800):
                    visualize_rollout(colours="linear")

            with not_linear_tab:
                with st.container(height=800):
                    visualize_rollout(colours="nonlinear")

        finally:
            del batch, pre_examples, pred, preds, att_weights, batched_rollout
            torch.cuda.empty_cache()

    with raw_att_tab:
        "Raw attention tab"
