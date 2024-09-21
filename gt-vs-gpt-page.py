import numpy as np
import streamlit as st
import torch
from streamlit_chat import message
import streamlit.components.v1 as components

from custom_data_utils.utils import create_grounding_annt_list, create_highlighted_passage
from visualization_data import init_model, InferenceDataProvider

EXAMPLE_VALIDATION_DATA = "data/examples/200_dialogues_reranking_gpt.json"

st.set_page_config(layout="wide")
chat, _, option1, option2 = st.columns([7, 1, 6, 6])


@st.cache_resource
def cache_init_model():
    return init_model()


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


@st.cache_data
def cache_dialog_inference_out(dialog_id):
    return data_provider.get_dialog_out(dialog_id)


cross_encoder, tokenizer = cache_init_model()
data_provider = InferenceDataProvider(cross_encoder, tokenizer)

# get available dialogues

# todo: filter out dialogues that have been annotated and are not available for annotation


set_data = {
    "current_dialogue": None,
    "gt_label_colour": "#2222DD",
    "dialogues_to_ann": data_provider.get_valid_dialog_ids()[20:]
}

if set_data["current_dialogue"] is None:
    new_random_dialogue(set_data)

diag_turns, grounded_agent_utterance, nr_show_utterances, rerank_dialog_examples \
    = cache_dialog_inference_out(set_data["current_dialogue"])

# MID SECTION CHAT
with chat:
    for utterance in diag_turns[:nr_show_utterances]:
        is_user = True if utterance["role"] == "user" else False
        message(f"{utterance['utterance']}", is_user=is_user)

    st.chat_input("Say something")


def example_preferred():
    pass


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


# Cross encoder inference on current dialogue
@st.cache_data
def cache_cross_encoder_inference(dialog_id, nr_passages):
    # Dialog id is used for caching management
    return data_provider.get_dialog_inference_out(dialog_id, nr_passages)


show_gt_examples = rerank_dialog_examples
show_gt_example = [example for example in show_gt_examples if example["label"] == 1][0]

with option1:
    with st.container(border=True):
        "### :red[Option A]"
        passage_list, gt_labels = create_grounding_annt_list(show_gt_example["passage"],
                                                             grounded_agent_utterance["references"],
                                                             show_gt_example["label"],
                                                             tokenizer)
        show_annotated_psg(passage_list, gt_label_list=gt_labels)

    neither_col, A_col = st.columns([50, 50])

with option2:
    with st.container(border=True):
        "### :green[Option B]"
        passage_list, gt_labels = create_grounding_annt_list(show_gt_example["passage"],
                                                             show_gt_example["gpt_references"],
                                                             show_gt_example["label"],
                                                             tokenizer)
        show_annotated_psg(passage_list, gt_label_list=gt_labels)

    # Add buttons to each column
    with neither_col:
        st.button("Neither is good", on_click=example_preferred, args=("None",))

    with A_col:
        st.button("A is better", on_click=example_preferred, args=("A",))

    B_col, both_col = st.columns([50, 50])

    with B_col:
        st.button('B is better', on_click=example_preferred, args=("B",))

    with both_col:
        st.button("Both are perfect", on_click=example_preferred, args=("Both",))

    change_button_colour('A is better', '', 'red')
    change_button_colour('B is better', '', 'green')

    add_justify_end_to_parent('A is better')
    add_justify_end_to_parent('Both are perfect')
