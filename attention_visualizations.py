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

# MID SECTION CHAT
with chat:
    message("Hello 👋", )
    message(f"{dialogue_index}", is_user=True)

    st.chat_input("Say something")

# RIGHT SECTION EXPLAINING features
with explaining:
    st.markdown("### Retrieved results and attention visualizations")
    with st.container(height=800):
        gt_tab, att_rollout_tab, raw_att_tab = st.tabs(["Ground Truth", "Attention Rollout", "Raw Attention"])

        with gt_tab:
            st.write("This is tab with ground truths will be displayed")
            st.write("This is a sidebar with a radio button")
            st.write("This is tab with ground truths will be displayed")
            st.write("This is a sidebar with a radio button")
            st.write("This is tab with ground truths will be displayed")
            st.write("This is a sidebar with a radio button")
            st.write("This is tab with ground truths will be displayed")
            st.write("This is a sidebar with a radio button")
            st.write("This is tab with ground truths will be displayed")
            st.write("This is a sidebar with a radio button")
            st.write("This is tab with ground truths will be displayed")
            st.write("This is a sidebar with a radio button")
            st.write("This is tab with ground truths will be displayed")
            st.write("This is a sidebar with a radio button")
            st.write("This is tab with ground truths will be displayed")
            st.write("This is a sidebar with a radio button")
            st.write("This is tab with ground truths will be displayed")
            st.write("This is a sidebar with a radio button")
            st.write("This is tab with ground truths will be displayed")
            st.write("This is a sidebar with a radio button")
            st.write("This is tab with ground truths will be displayed")
            st.write("This is a sidebar with a radio button")
            st.write("This is tab with ground truths will be displayed")
            st.write("This is a sidebar with a radio button")
            st.write("This is tab with ground truths will be displayed")
            st.write("This is a sidebar with a radio button")
            st.write("This is tab with ground truths will be displayed")
            st.write("This is a sidebar with a radio button")

    with att_rollout_tab:
        st.write("Att rollout tab")

    with raw_att_tab:
        st.write("Raw attention tab")
