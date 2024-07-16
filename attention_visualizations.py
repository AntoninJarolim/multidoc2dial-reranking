import json

import numpy as np
import streamlit as st
from streamlit_chat import message

st.set_page_config(layout="wide")

# Layout config
chat, _, explaining = st.columns([6, 1, 6])
data = np.random.randn(10, 1)

# DATA
EXAMPLE_VALIDATION_DATA = "data/examples/200_dialogues_reranking.json"


@st.cache_data
def get_data():
    return json.load(open(EXAMPLE_VALIDATION_DATA))


def split_utterance_history(raw_rarank_data):
    return list(map(lambda x: {
        "x": x["x"],
        "label": x["label"],
        "utterance_history": x["x"].split("[SEP]")[0],
        "passage": x["x"].split("[SEP]")[1]
    }, raw_rarank_data))


data_dialogues = get_data()
print(data_dialogues[0])

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

selected_dialog = data_dialogues[dialogue_index]
diag_turns = selected_dialog["dialog"]["turns"]
rerank_dialog_examples = split_utterance_history(selected_dialog["to_rerank"])
utterance_history, passage = [(d["utterance_history"], d["passage"])
                              for d in rerank_dialog_examples
                              if d["label"]][0]
last_user_utterance = utterance_history.split("agent: ")[0]
last_user_utterance_id = [t['turn_id'] for t in diag_turns if t["utterance"] == last_user_utterance][0]
grounded_agent_utterance_id = last_user_utterance_id + 1
grounded_agent_utterance = diag_turns[grounded_agent_utterance_id]
nr_show_utterances = grounded_agent_utterance_id

# MID SECTION CHAT
with chat:
    for utterance in diag_turns[:nr_show_utterances]:
        is_user = True if utterance["role"] == "user" else False
        message(f"{utterance['utterance']}", is_user=is_user)

    st.chat_input("Say something")


# RIGHT SECTION EXPLAINING features
def show_annotated_psg(passage_text, idx, is_grounding=False):
    with st.container(border=True):
        col_idx, col2 = st.columns([6, 100])
        with col_idx:
            f'##### {idx}'

        with col2:
            f"{passage_text}"
        # annotated_text(
        #     (passage_text, "", set_data["gt_label_colour"]),
        # )


with explaining:
    "### Reranked results and attention visualizations"
    with st.container(height=800):
        gt_tab, att_rollout_tab, raw_att_tab = st.tabs(["Ground Truth", "Attention Rollout", "Raw Attention"])

        with gt_tab:
            gt_label = "GT"

            for i, example in enumerate(rerank_dialog_examples, start=1):
                show_annotated_psg(example["passage"], i, example["label"])

        with att_rollout_tab:
            "Att rollout tab"

        with raw_att_tab:
            "Raw attention tab"
