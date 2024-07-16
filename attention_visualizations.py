from contextlib import contextmanager

import numpy as np
import streamlit as st
from streamlit_chat import message

st.set_page_config(layout="wide")

chat, _, explaining = st.columns([6, 1, 6])
data = np.random.randn(10, 1)


# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )

print(add_radio)

with chat:
    message("Hello 👋", )
    message(f"{add_radio}", is_user=True)

    st.chat_input("Say something")

with explaining:
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

