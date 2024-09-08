import numpy as np
import streamlit as st
import torch
from streamlit_chat import message

from custom_data_utils.utils import create_grounding_annt_list, create_highlighted_passage
from visualization_data import init_model, InferenceDataProvider

st.set_page_config(layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Layout config
chat, _, explaining = st.columns([6, 1, 6])
data = np.random.randn(10, 1)

# DATA
EXAMPLE_VALIDATION_DATA = "data/examples/200_dialogues_reranking_gpt.json"
model_name = "naver/trecdl22-crossencoder-debertav3"


@st.cache_resource
def cache_init_model():
    return init_model()


cross_encoder, tokenizer = cache_init_model()


def show_annotated_psg(passage_tokens, idx, is_grounding=False, annotation_scores=None, base_colour="blue",
                       colour_type="linear", score=None, hide_attention_colours=False, gt_label_list=None):
    """
    :param gt_label_list: list of True/False, True - GT word, False - no GT
    :param passage_tokens: list of strings of same length as annotation_sc
    :param annotation_scores: list of numbers used to colour the passage_text
    :param idx: ID of the passage to show
    :param is_grounding: Whether the passage is the grounding passage - 'id' will be coloured blue
    """
    assert not isinstance(passage_tokens, str)

    with st.container(border=True):
        col_idx, passage_col = st.columns([77, 1000])
        with col_idx:
            # Print psg index number in colour if grounding
            f'### :blue[{idx}]' if is_grounding else f'### {idx}'

        with passage_col:
            if score is not None:
                f"#### {score}"

            if hide_attention_colours:
                annotation_scores = None

            highlighted_passage = create_highlighted_passage(passage_tokens, gt_label_list, annotation_scores,
                                                             base_colour, colour_type)
            st.html("\n".join(highlighted_passage))


# This variables will be set in configuration sidebar
set_data = {
    "current_dialogue": 0,
    "gt_label_colour": "#2222DD",
    "show_explanations": False,
    "show_relevance_score": False,
    "hide_attention_colours": False,
    "online_inference": False,
    "show_raw_attention": False,
    "nr_passages": 16
}

data_provider = InferenceDataProvider(cross_encoder, tokenizer)

# CONFIGURATION SIDEBAR
with st.sidebar:
    "## Configuration"
    "### Dialog loading"
    set_data["online_inference"] = st.toggle("Online inference", set_data["online_inference"])
    choose_by_relevance = st.toggle("Sorted by relevance", False)
    dialogue_index = st.selectbox('Dialog id:',
                                  data_provider.get_valid_dialog_ids(), key="dialogue_index",
                                  index=set_data["current_dialogue"],
                                  disabled=choose_by_relevance)

    dialogue_index_relevance = st.selectbox('Dialog id by relevance score:',
                                            data_provider.get_inf_sorted_dialog_ids(), key="dialogue_index_relevance",
                                            index=set_data["current_dialogue"],
                                            disabled=not choose_by_relevance)

    set_data["current_dialogue"] = dialogue_index_relevance if choose_by_relevance else dialogue_index

    "### Visualization settings"
    set_data["nr_passages"] = st.selectbox('Number of passages to show:',
                                           list(range(32)),
                                           index=set_data["nr_passages"])
    set_data["show_explanations"] = st.toggle("Show explanations", set_data["show_explanations"])
    set_data["show_relevance_score"] = st.toggle("Show relevance score", set_data["show_relevance_score"])
    set_data["hide_attention_colours"] = st.toggle("Hide attention colours", set_data["hide_attention_colours"])
    set_data["show_raw_attention"] = st.toggle("Show raw attention", set_data["show_raw_attention"])

data_provider_mode = "online" if set_data["online_inference"] else "offline"
data_provider.set_mode(data_provider_mode)


@st.cache_data
def cache_dialog_inference_out(dialog_id):
    return data_provider.get_dialog_out(dialog_id)


diag_turns, grounded_agent_utterance, nr_show_utterances, rerank_dialog_examples \
    = cache_dialog_inference_out(set_data["current_dialogue"])

# MID SECTION CHAT
with chat:
    for utterance in diag_turns[:nr_show_utterances]:
        is_user = True if utterance["role"] == "user" else False
        message(f"{utterance['utterance']}", is_user=is_user)

    st.chat_input("Say something")


# Cross encoder inference on current dialogue
@st.cache_data
def cache_cross_encoder_inference(dialog_id, nr_passages):
    # Dialog id is used for caching management
    return data_provider.get_dialog_inference_out(dialog_id, nr_passages)


# RIGHT SECTION EXPLAINING features
with (explaining):
    "### Reranked results and attention visualizations"
    gt_tab, gpt_tab, raw_att_tab, att_rollout_tab, grad_sam_tab, att_cat_tab = st.tabs(
        ["Ground Truth",
         "gpt-4o-2024-08-06",
         "Raw Attention",
         "Attention Rollout",
         "Grad-SAM",
         "Att-CAT",
         ])

    show_gt_examples = rerank_dialog_examples[:set_data["nr_passages"]]

    with gt_tab:
        with st.container(height=800):
            for i, example in enumerate(show_gt_examples, start=1):
                passage_list, gt_labels = create_grounding_annt_list(example["passage"],
                                                                     grounded_agent_utterance["references"],
                                                                     example["label"],
                                                                     tokenizer)
                show_annotated_psg(passage_list, i, example["label"] == 1, gt_label_list=gt_labels)

    inf_out = cache_cross_encoder_inference(set_data["current_dialogue"], set_data["nr_passages"])
    show_gt_examples = inf_out["reranked_examples"]

    with gpt_tab:
        with st.container(height=800):
            # Set grounding to 1 for all passages (using same function as for GT)
            is_grounding = 1
            for i, example in enumerate(show_gt_examples, start=1):
                if "gpt_references" in example:
                    passage_list, gt_labels = create_grounding_annt_list(example["passage"],
                                                                         grounded_agent_utterance["references"],
                                                                         example["label"],
                                                                         tokenizer)
                    _, gpt_labels_refs, failed_refs = create_grounding_annt_list(example["passage"],
                                                                                 example["gpt_references"],
                                                                                 is_grounding,
                                                                                 tokenizer,
                                                                                 return_failed=True)
                    if failed_refs:
                        f"##### Parsing of GPT references failed for passage {i}\n"
                        "Failed references: ", failed_refs, "\n"
                        # create_grounding_annt_list(example["passage"],
                        #                            example["gpt_references"],
                        #                            is_grounding)
                    # f"##### references json for passage id: {i}\n"
                    # example["gpt_references"]

                    show_annotated_psg(passage_list, i, example["label"] == 1, gt_label_list=gt_labels,
                                       base_colour="green", annotation_scores=gpt_labels_refs)
            else:
                with st.container(border=True):
                    "#### GPT references not generated for this passage! \n"
                    example["passage"]


    def visualize_example(scores_to_colour, colour_type="linear"):
        for idx, (r_example, score_colour, score) in enumerate(
                zip(inf_out['reranked_examples'], scores_to_colour, inf_out['sorted_predictions']),
                start=1):
            score = score if set_data["show_relevance_score"] else None
            passage_list, gt_labels = create_grounding_annt_list(r_example['x'],
                                                                 grounded_agent_utterance["references"],
                                                                 r_example["label"],
                                                                 tokenizer)
            score_colour = score_colour[1:][:-1][:len(passage_list)]
            show_annotated_psg(passage_list, idx, is_grounding=r_example["label"], annotation_scores=score_colour,
                               base_colour="green", colour_type=colour_type, score=score,
                               hide_attention_colours=set_data["hide_attention_colours"],
                               gt_label_list=gt_labels)


    def lin_non_lin_tab(scores_to_colour):
        linear_tab, not_linear_tab = st.tabs(["Linear colour", "Non-linear colours"])
        with linear_tab:
            with st.container(height=800):
                visualize_example(scores_to_colour, colour_type="linear")

        with not_linear_tab:
            with st.container(height=800):
                visualize_example(scores_to_colour, colour_type="nonlinear")


    with raw_att_tab:
        if set_data["show_raw_attention"]:
            l_idx = st.number_input("Raw Attention Layer", min_value=1, max_value=24) - 1
            att_weights = inf_out["att_weights_cls"]  # (L, B, H, S)

            head_tabs = st.tabs([f"Head {i + 1}" for i in range(att_weights[0].shape[1])])
            for h_idx, head_tab in enumerate(head_tabs):
                with head_tab:
                    att_map = [w[l_idx, h_idx, :] for w in att_weights]
                    lin_non_lin_tab(att_map)
        else:
            "#### Raw attention is disabled"

    with att_rollout_tab:
        lin_non_lin_tab(inf_out['reranked_rollouts'])

    with grad_sam_tab:
        lin_non_lin_tab(inf_out['grad_sam_scores'])

    with att_cat_tab:
        lin_non_lin_tab(inf_out['att_cat_scores'])

    # with grad_sam_tab_spans:
    #     lin_non_lin_tab(inf_out['grad_sam_scores_spans'])
    # with grad_sam_tab_spans2:
    #     lin_non_lin_tab(inf_out['grad_sam_scores_spans_K_N'])
