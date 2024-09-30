import json
import os
import random
from collections import defaultdict
from datetime import datetime
from os.path import join, dirname

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit_chat import message

from custom_data_utils.utils import create_grounding_annt_list, create_highlighted_passage
from visualization_data import init_model, InferenceDataProvider

EXAMPLE_VALIDATION_DATA = "data/examples/200_dialogues_reranking_gpt.json"

st.set_page_config(
    layout="wide",
    page_title="A/B Testing",
)

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

DEBUG = os.environ.get("DEBUG", False)
DEBUG = True if DEBUG == "True" else False
load_time = datetime.now()


def get_remote_ip() -> str:
    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return ""

        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return ""
    except:
        return ""

    return session_info.request.remote_ip


@st.cache_resource
def cache_init_model():
    return init_model(tokenizer_only=True)


@st.cache_data
def cache_dialog_inference_out(dialog_id):
    return data_provider.get_dialog_out(dialog_id)


def change_button_colour(widget_label, font_color='', background_color='transparent'):
    htmlstr = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{ 
                if (elements[i].innerText == '{widget_label}') {{ 
                    elements[i].style.color ='{font_color}';
                    elements[i].style.background = '{background_color}'
                }}
            }}
        </script>
        """
    components.html(f"{htmlstr}")


def add_justify_end_to_parent(widget_label):
    htmlstr = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{
                if (elements[i].innerText == '{widget_label}') {{
                    // Add the style to the parent of the button
                    elements[i].parentElement.style.display = 'flex';
                    elements[i].parentElement.style.justifyContent = 'end';
                }}
            }}
        </script>
        """
    components.html(f"{htmlstr}")


def read_done_dialogues():
    done_dialogues = defaultdict(int)
    if os.path.exists("user-choices/user-preferences.jsonl"):
        with open("user-choices/user-preferences.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                done_dialogues[data["dialogue_id"]] += 1
    return done_dialogues


def ids_with_highest_annotations():
    freq_dict = read_done_dialogues()
    if len(freq_dict) == 0:
        return []
    max_value = max(freq_dict.values())
    return [k for k, v in freq_dict.items() if v == max_value]


def new_random_dialogue(set_data):
    all_choices = set(set_data["dialogues_to_ann"])
    choices = set(ids_with_highest_annotations())
    set_data["current_dialogue"] = np.random.choice(list(all_choices - choices))


def show_annotated_psg(passage_tokens, annotation_scores=None, base_colour="blue",
                       colour_type="linear", score=None, hide_attention_colours=False, gt_label_list=None):
    """
    :param gt_label_list: list of True/False, True - GT word, False - no GT
    :param passage_tokens: list of strings of same length as annotation_sc
    :param annotation_scores: list of numbers used to colour the passage_text
    """
    assert not isinstance(passage_tokens, str)
    if score is not None:
        f"#### {score}"

    if hide_attention_colours:
        annotation_scores = None

    highlighted_passage = create_highlighted_passage(passage_tokens, gt_label_list, annotation_scores,
                                                     base_colour, colour_type)
    st.html("\n".join(highlighted_passage))


def example_preferred(preference):
    if os.environ.get("DEBUG", False):
        print(f"Preference: {preference}")

    if not os.path.exists("user-choices"):
        os.makedirs("user-choices")

    with open(f"user-choices/user-preferences.jsonl", "a") as f:
        out_obj = {
            "dialogue_id": str(set_data["current_dialogue"]),
            "preference": preference,
            "time": str(datetime.now()),
            "took_seconds": (datetime.now() - load_time).total_seconds(),
            "ip": get_remote_ip()
        }
        out_data = json.dumps(out_obj)
        print(out_data)
        f.write(f"{out_data}\n")


def get_gt_passage(examples):
    return [example for example in examples if example["label"] == 1][0]


def filter_grounded_passage_has_gpt_refs():
    dialogue_ids = data_provider.get_valid_dialog_ids()
    has_ids = []
    for dialogue_id in dialogue_ids:
        _, _, _, all_examples = data_provider.get_dialog_out(dialogue_id)
        gt_psg = get_gt_passage(all_examples)
        if "gpt_references" in gt_psg:
            has_ids.append(dialogue_id)

    excluded_ids = [36]
    return [dialogue_id for dialogue_id in has_ids if dialogue_id not in excluded_ids]


# Start of streamlit page execution
tokenizer = cache_init_model()
data_provider = InferenceDataProvider()

# get available dialogues

# todo: filter out dialogues that have been annotated and are not available for annotation

# Setting which data to show
set_data = {
    "current_dialogue": None,
    "gt_label_colour": "#2222DD",
    "dialogues_to_ann": filter_grounded_passage_has_gpt_refs()[:30]
}

# Initialize session state
if 'next_dialogue_id' not in st.session_state:
    st.session_state.next_dialogue_id = None

if set_data["current_dialogue"] is None:
    if st.session_state.next_dialogue_id:
        # Reset the dialogue if the admin has manually changed it
        set_data["current_dialogue"] = st.session_state.next_dialogue_id
        st.session_state.next_dialogue_id = None
    else:
        new_random_dialogue(set_data)

diag_turns, grounded_agent_utterance, nr_show_utterances, rerank_dialog_examples \
    = cache_dialog_inference_out(set_data["current_dialogue"])


def skip_dialogue():
    current_id = [i for i, diag_íd in enumerate(set_data["dialogues_to_ann"])
                  if diag_íd == set_data["current_dialogue"]][0]
    next_id = current_id + 1
    if next_id >= len(set_data["dialogues_to_ann"]):
        next_id = 0
    st.session_state.next_dialogue_id = set_data["dialogues_to_ann"][next_id]


if DEBUG:
    with st.sidebar:
        st.write("Debug mode")
        st.write("Current dialogue:", set_data["current_dialogue"])
        "diags to annotate", set_data["dialogues_to_ann"]
        st.button("Next dialogue", on_click=skip_dialogue)
        "digs with highest annotations", ids_with_highest_annotations()

chat, _, option1, option2 = st.columns([7, 1, 6, 6])
# Left side of the page
with chat:
    for utterance in diag_turns[:nr_show_utterances - 1]:
        is_user = True if utterance["role"] == "user" else False
        message(f"{utterance['utterance']}", is_user=is_user)

    st.chat_input("Say something")

show_gt_examples = rerank_dialog_examples
show_gt_example = get_gt_passage(show_gt_examples)

A_passage, A_labels = create_grounding_annt_list(show_gt_example["passage"],
                                                 grounded_agent_utterance["references"],
                                                 show_gt_example["label"],
                                                 tokenizer)

B_passage, B_labels = create_grounding_annt_list(show_gt_example["passage"],
                                                 show_gt_example["gpt_references"],
                                                 show_gt_example["label"],
                                                 tokenizer)

left_label = "GT"
right_label = "GPT"
if random.random() < 0.5:
    A_passage, B_passage = B_passage, A_passage
    A_labels, B_labels = B_labels, A_labels
    left_label, right_label = right_label, left_label

with option1:
    with st.container(border=True):
        "### :red[Option A]"
        if DEBUG:
            f"### {left_label}"
        show_annotated_psg(A_passage, gt_label_list=A_labels)

    neither_col, A_mid_col, A_col = st.columns([50, 50, 50])

with option2:
    with st.container(border=True):
        "### :green[Option B]"
        if DEBUG:
            f"### {right_label}"
        show_annotated_psg(B_passage, gt_label_list=B_labels)

    B_col, B_mid_col, both_col = st.columns([50, 50, 50])

# Add buttons to each column
with neither_col:
    st.button("Neither is good", on_click=example_preferred, args=("None",))

with A_mid_col:
    st.button("A slightly better", on_click=example_preferred, args=(left_label + "-slightly",))

with A_col:
    st.button("A better", on_click=example_preferred, args=(left_label,))

with B_col:
    st.button('B better', on_click=example_preferred, args=(right_label,))

with B_mid_col:
    st.button('B slightly better', on_click=example_preferred, args=(right_label + "-slightly",))

with both_col:
    st.button("Both are perfect", on_click=example_preferred, args=("Both",))

change_button_colour('A better', background_color='#e51f1f')
change_button_colour('B better', background_color='#70e000')

change_button_colour('A slightly better', background_color='#e51f1f99')
change_button_colour('B slightly better', background_color='#70e00099')

add_justify_end_to_parent('A better')
add_justify_end_to_parent('Both are perfect')
