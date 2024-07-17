import json

import numpy as np
import streamlit as st
import torch
from annotated_text import annotated_text
from streamlit_chat import message
from transformers import AutoTokenizer

import utils
from train_ce import CrossEncoder

st.set_page_config(layout="wide")

# Layout config
chat, _, explaining = st.columns([6, 1, 6])
data = np.random.randn(10, 1)

# DATA
EXAMPLE_VALIDATION_DATA = "data/examples/200_dialogues_reranking.json"
model_name = "naver/trecdl22-crossencoder-debertav3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
cross_encoder = CrossEncoder(model_name)
model = utils.load_model(cross_encoder, "CE_lr0.00010485388296357131_bs16.pt")


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


data_dialogues = get_data()

# This variables will be set in configuration sidebar
set_data = {
    "current_dialogue": 0,
    "gt_label_colour": "#2222DD",
}

# CONFIGURATION SIDEBAR
with st.sidebar:
    "## Configuration"
    "### Dialog loading"
    dialogue_index = st.selectbox('Example dialog id:', list(range(len(data_dialogues))))
    set_data["current_dialogue"] = dialogue_index

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
def annt_list_2_colours(annotation_list, base_colour):
    assert base_colour in ["blue", "red", "green"], f"Base colour {base_colour} not supported"

    # Normalize annotation list and conver to ints (0-255)
    tensor_list = torch.tensor(annotation_list)
    normalized_tensor_list = tensor_list / torch.sum(tensor_list)
    normalized_tensor_list = normalized_tensor_list / torch.max(normalized_tensor_list)
    colour_range = (normalized_tensor_list * 255).type(torch.int64)

    assert torch.max(colour_range) < 256 and 0 >= torch.min(colour_range), "Conversion to colour range failed"

    if base_colour == "blue":
        def conv_fce(x):
            return f'#1111{x:02x}'
    elif base_colour == "green":
        def conv_fce(x):
            return f'#00{x:02x}00'
    elif base_colour == "red":
        def conv_fce(x):
            return f'#{x:02x}0000'
    else:
        raise ValueError(f"Base colour {base_colour} not supported")

    coloured_list = [conv_fce(x) for x in colour_range]
    return [x if x not in ["#000000", "#111100"] else None
            for x in coloured_list]


def show_annotated_psg(passage_text, idx, is_grounding=False, annotation_list=None):
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
            if annotation_list:
                colours_annotation_list = annt_list_2_colours(annotation_list, "blue")
                coloured_passage = []
                for colour, text in zip(colours_annotation_list, passage_text):
                    if colour is None:
                        coloured_passage.append(text)
                    else:
                        text_tokens = split_to_tokens(text)
                        for token in text_tokens:
                            coloured_passage.append((token, "", colour))

                annotated_text(coloured_passage)
            else:
                passage_text
        # annotated_text
        #     (passage_text, "", set_data["gt_label_colour"]),
        # )


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

        try:
            before, after = passage.split(ref_span)
        except ValueError:
            print(f"Passage: {passage}")
            print(f"Ref span: {ref_span}")
            exit(6)

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
    with st.container(height=800):
        gt_tab, att_rollout_tab, raw_att_tab = st.tabs(["Ground Truth", "Attention Rollout", "Raw Attention"])

        with gt_tab:
            gt_label = "GT"

            for i, example in enumerate(rerank_dialog_examples, start=1):
                passage_list, annotation_list = create_grounding_annt_list(example["passage"],
                                                                           grounded_agent_utterance,
                                                                           example["label"])
                show_annotated_psg(passage_list, i, example["label"], annotation_list)

        with att_rollout_tab:


        with raw_att_tab:
            "Raw attention tab"
