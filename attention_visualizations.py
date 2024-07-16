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


data_dialogues = get_data()
print(data_dialogues[0])

# This variables will be set in configuration sidebar
set_data = {
    "current_dialogue": 0,
}

# CONFIGURATION SIDEBAR
with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("### Dialog loading")
    dialogue_index = st.selectbox('Example dialog id:', list(range(len(data_dialogues))))
    set_data["current_dialogue"] = dialogue_index

selected_dialog = data_dialogues[dialogue_index]
diag_turns = selected_dialog["dialog"]["turns"]
rerank_dialog_examples = selected_dialog["to_rerank"]
utterance_history, passage = [d["x"] for d in rerank_dialog_examples if d["label"]][0].split("[SEP]")
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
with explaining:
    st.markdown("### Reranked results and attention visualizations")
    with st.container(height=800):
        gt_tab, att_rollout_tab, raw_att_tab = st.tabs(["Ground Truth", "Attention Rollout", "Raw Attention"])

        with gt_tab:
            st.write("This is tab with ground truths will be displayed")
            st.write("This is a sidebar with a radio button")

    with att_rollout_tab:
        st.write("Att rollout tab")

    with raw_att_tab:
        st.write("Raw attention tab")
