import os
from os.path import join, dirname

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from streamlit_chat import message

from custom_data_utils.utils import create_grounding_annt_list, create_highlighted_passage
from visualization_data import init_model, InferenceDataProvider

EXAMPLE_VALIDATION_DATA = "data/examples/200_dialogues_reranking_gpt.json"

st.set_page_config(layout="wide")
chat, _, option1, option2 = st.columns([7, 1, 6, 6])

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

DEBUG = os.environ.get("DEBUG", False)


@st.cache_resource
def cache_init_model():
    return init_model()


@st.cache_data
def cache_dialog_inference_out(dialog_id):
    return data_provider.get_dialog_out(dialog_id)


def change_button_colour(widget_label, font_color, background_color='transparent'):
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


def new_random_dialogue(set_data):
    set_data["current_dialogue"] = np.random.choice(set_data["dialogues_to_ann"])


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


def example_preferred():
    pass


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
    return has_ids


# Start of streamlit page execution
cross_encoder, tokenizer = cache_init_model()
data_provider = InferenceDataProvider(cross_encoder, tokenizer)

# get available dialogues

# todo: filter out dialogues that have been annotated and are not available for annotation

# Setting which data to show
set_data = {
    "current_dialogue": None,
    "gt_label_colour": "#2222DD",
    "dialogues_to_ann": filter_grounded_passage_has_gpt_refs()[:30]
}

if set_data["current_dialogue"] is None:
    new_random_dialogue(set_data)

diag_turns, grounded_agent_utterance, nr_show_utterances, rerank_dialog_examples \
    = cache_dialog_inference_out(set_data["current_dialogue"])

if DEBUG:
    with st.sidebar:
        st.write("Debug mode")
        st.write("Current dialogue:", set_data["current_dialogue"])
        "diags to annotate", set_data["dialogues_to_ann"]
        st.button("Next dialogue", on_click=new_random_dialogue, args=(set_data,))

# Left side of the page
with chat:
    for utterance in diag_turns[:nr_show_utterances]:
        is_user = True if utterance["role"] == "user" else False
        message(f"{utterance['utterance']}", is_user=is_user)

    st.chat_input("Say something")


def skip_dialogue():
    set_data["current_dialogue"] += 1


show_gt_examples = rerank_dialog_examples
show_gt_example = get_gt_passage(show_gt_examples)

with option1:
    with st.container(border=True):
        "### :red[Option A]"
        passage_list, gt_labels = create_grounding_annt_list(show_gt_example["passage"],
                                                             grounded_agent_utterance["references"],
                                                             show_gt_example["label"],
                                                             tokenizer)
        show_annotated_psg(passage_list, gt_label_list=gt_labels)

    neither_col, A_mid_col, A_col = st.columns([50, 50, 50])

with option2:
    with st.container(border=True):
        "### :green[Option B]"
        passage_list, gt_labels = create_grounding_annt_list(show_gt_example["passage"],
                                                             show_gt_example["gpt_references"],
                                                             show_gt_example["label"],
                                                             tokenizer)
        show_annotated_psg(passage_list, gt_label_list=gt_labels)

    B_col, B_mid_col, both_col = st.columns([50, 50, 50])

# Add buttons to each column
with neither_col:
    st.button("Neither is good", on_click=example_preferred, args=("None",))

with A_mid_col:
    st.button("A slightly better", on_click=example_preferred, args=("A",))

with A_col:
    st.button("A better", on_click=example_preferred, args=("A",))

with B_col:
    st.button('B better', on_click=example_preferred, args=("B",))

with B_mid_col:
    st.button('B slightly better', on_click=example_preferred, args=("B",))

with both_col:
    st.button("Both are perfect", on_click=example_preferred, args=("Both",))

change_button_colour('A better', '', '#e51f1f')
change_button_colour('B better', '', '#70e000')

change_button_colour('A slightly better', '', '#e51f1f99')
change_button_colour('B slightly better', '', '#70e00099')

add_justify_end_to_parent('A better')
add_justify_end_to_parent('Both are perfect')
