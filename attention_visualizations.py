import numpy as np
import streamlit as st
from streamlit_chat import message

st.set_page_config(layout="wide")

chat, _, explaining = st.columns([3, 1, 2])
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
    with st.container(border=True):
        st.write("This is inside the container")

    st.write("This is a sidebar with a radio button")
