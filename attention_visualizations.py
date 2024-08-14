import numpy as np
import streamlit as st
import torch
from streamlit_chat import message

from visualization_data import init_model, InferenceDataProvider

st.set_page_config(layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Layout config
chat, _, explaining = st.columns([6, 1, 6])
data = np.random.randn(10, 1)

# DATA
EXAMPLE_VALIDATION_DATA = "data/examples/200_dialogues_reranking.json"
model_name = "naver/trecdl22-crossencoder-debertav3"


@st.cache_resource
def cache_init_model():
    return init_model()


cross_encoder, tokenizer = cache_init_model()


def split_to_tokens(text):
    return tokenizer.batch_decode(tokenizer(text)["input_ids"])[1:][:-1]


def annt_list_2_colours(annotation_list, base_colour, colours):
    if annotation_list is None:
        return None
    assert base_colour in ["blue", "red", "green"], f"Base colour {base_colour} not supported"

    # Normalize annotation list and convert to ints (0-255)
    if not isinstance(annotation_list, torch.Tensor):
        annotation_list = torch.Tensor(annotation_list)

    normalized_tensor_list = annotation_list / torch.max(annotation_list)
    if colours == "nonlinear":
        transf_list = -torch.log(normalized_tensor_list)
        normalized_tensor_list = 1 - (transf_list / torch.max(transf_list))

    colour_range = (normalized_tensor_list * 255).type(torch.int64)

    if not (0 <= torch.min(colour_range)):
        f"min: Conversion of {torch.min(colour_range)} to colour range failed"
    assert torch.max(colour_range) < 256, f"max: Conversion of {torch.max(colour_range)} to colour range failed"

    if base_colour == "blue":
        def conv_fce(x):
            return f'#1111{x:02x}'
    elif base_colour == "green":
        def conv_fce(x):
            return f'#11{x:02x}11'
    elif base_colour == "red":
        def conv_fce(x):
            return f'#{x:02x}0000'
    else:
        raise ValueError(f"Base colour {base_colour} not supported")

    coloured_list = [conv_fce(x) for x in colour_range]
    return [x if x not in ["#000000", "#111100", "#11011"] else ["#00000000"]
            for x in coloured_list]


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

        # Create default list of colours for each token
        colours_annotation_list = ["#00000000"] * len(passage_tokens)
        if annotation_scores is not None and not hide_attention_colours:
            colours_annotation_list = annt_list_2_colours(annotation_scores, base_colour, colour_type)

        if gt_label_list is None:
            gt_label_list = [None] * len(passage_tokens)

        def display_colored_word(word, bg_color, fg_color):
            # Create the HTML string with inline CSS for styling
            return f"""
                <span style="display: 
                inline-flex; 
                flex-direction: row; 
                align-items: center; 
                background: {bg_color}; 
                border-radius: 0.5rem; 
                padding: 0.25rem 0.5rem; 
                overflow: hidden; 
                line-height: 1; 
                color: {fg_color};
                ">                
                {word}
                </span>
            """

        with passage_col:
            if score is not None:
                f"#### {score}"

            psg_custom = []
            for bg_colour, token, gt_label in zip(colours_annotation_list, passage_tokens, gt_label_list):
                fg_colour = "#FFFFFF" if not gt_label else "#4455FF"
                token = token.replace('$', '\$')
                span_text = display_colored_word(token, bg_colour, fg_colour)
                psg_custom.append(span_text)

            st.html("\n".join(psg_custom))


def create_grounding_annt_list(passage, grounded_agent_utterance, label):
    if label == 0:
        return split_to_tokens(passage), None

    # Create annotation list
    annotation_list = []
    broken_passage = []
    for reference in grounded_agent_utterance["references"]:
        # strip space, because it is sometimes appended at the end and the space is not in
        # the passage, leading to not finding part of passage containing this reference span
        ref_span = reference["ref_span"].strip(" ")

        if passage == ref_span:
            annotation_list.append(True)
            broken_passage.append(ref_span)
            break

        before, after = passage.split(ref_span)

        # Found reference is not at the beginning of the passage
        if before != "":
            annotation_list.append(False)
            broken_passage.append(before)  # Do not label this part of passage as GT

        annotation_list.append(True)
        broken_passage.append(ref_span)

        passage = after
        if passage == "":
            break

    # Append the remaining part of passage (if any)
    if passage != "":
        annotation_list.append(False)
        broken_passage.append(passage)

    unpack_passages = []
    unpack_annt_list = []
    for psg, annt in zip(broken_passage, annotation_list):
        psg_tokens = split_to_tokens(psg)
        for t in psg_tokens:
            unpack_passages.append(t)
            unpack_annt_list.append(annt)

    return unpack_passages, unpack_annt_list


# This variables will be set in configuration sidebar
set_data = {
    "current_dialogue": 0,
    "gt_label_colour": "#2222DD",
    "show_explanations": False,
    "show_relevance_score": False,
    "hide_attention_colours": False,
    "online_inference": False
}

data_provider = InferenceDataProvider(cross_encoder, tokenizer)

# CONFIGURATION SIDEBAR
with st.sidebar:
    "## Configuration"
    "### Dialog loading"
    dialogue_index = st.selectbox('Example dialog id:',
                                  list(range(data_provider.get_nr_dialogs())), key="dialogue_index",
                                  index=set_data["current_dialogue"])
    set_data["current_dialogue"] = dialogue_index

    "### Togglers"
    set_data["show_explanations"] = st.toggle("Show explanations", set_data["show_explanations"])
    set_data["show_relevance_score"] = st.toggle("Show relevance score", set_data["show_relevance_score"])
    set_data["hide_attention_colours"] = st.toggle("Hide attention colours", set_data["hide_attention_colours"])
    set_data["online_inference"] = st.toggle("Online inference", set_data["online_inference"])

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
def cache_cross_encoder_inference(dialog_id):
    return data_provider.get_dialog_inference_out(dialog_id)


# RIGHT SECTION EXPLAINING features
with (explaining):
    "### Reranked results and attention visualizations"
    gt_tab, raw_att_tab, att_rollout_tab, grad_sam_tab = st.tabs(["Ground Truth",
                                                                  "Raw Attention",
                                                                  "Attention Rollout",
                                                                  "Grad-SAM"])

    with gt_tab:
        with st.container(height=800):
            for i, example in enumerate(rerank_dialog_examples, start=1):
                passage_list, gt_labels = create_grounding_annt_list(example["passage"],
                                                                     grounded_agent_utterance,
                                                                     example["label"])
                show_annotated_psg(passage_list, i, example["label"], gt_label_list=gt_labels)

    inf_out = cache_cross_encoder_inference(rerank_dialog_examples)


    def visualize_example(scores_to_colour, colour_type="linear"):
        for idx, (r_example, score_colour, score) in enumerate(
                zip(inf_out['reranked_examples'], scores_to_colour, inf_out['sorted_predictions']),
                start=1):
            score = score if set_data["show_relevance_score"] else None
            passage_list, gt_labels = create_grounding_annt_list(r_example['x'],
                                                                 grounded_agent_utterance,
                                                                 r_example["label"])
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
        l_idx = st.number_input("Raw Attention Layer", min_value=1, max_value=24) - 1
        att_weights = inf_out["att_weights_cls"]  # (L, B, H, S)

        head_tabs = st.tabs([f"Head {i + 1}" for i in range(att_weights[0].shape[1])])
        for h_idx, head_tab in enumerate(head_tabs):
            with head_tab:
                att_map = [w[l_idx, h_idx, :] for w in att_weights]
                lin_non_lin_tab(att_map)

    with att_rollout_tab:
        lin_non_lin_tab(inf_out['reranked_rollouts'])

    with grad_sam_tab:
        lin_non_lin_tab(inf_out['grad_sam_scores'])
