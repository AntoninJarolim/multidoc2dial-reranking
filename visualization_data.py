import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from custom_data_utils import utils
from custom_data_utils.utils import split_to_tokens
from interpretability import attention_rollout, grad_sam, att_cat
from md2d_dataset import preprocess_examples
from train_ce import CrossEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXAMPLE_VALIDATION_DATA = "data/examples/200_dialogues_reranking.json"
model_name = "naver/trecdl22-crossencoder-debertav3"


def init_model(tokenizer_only=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer_only:
        return tokenizer
    cross_encoder = CrossEncoder(model_name)
    cross_encoder.save_attention_weights = True
    cross_encoder.bert_model.config.output_attentions = True
    utils.load_model(cross_encoder, "CE_lr0.00010485388296357131_bs16.pt")
    cross_encoder.to(device)
    cross_encoder.eval()
    return cross_encoder, tokenizer


def get_data():
    return json.load(open(EXAMPLE_VALIDATION_DATA))


def split_utterance_history(raw_rarank_data):
    return list(map(lambda x: {
        "x": x["x"],
        "label": x["label"],
        "utterance_history": x["x"].split("[SEP]")[0],
        "passage": x["x"].split("[SEP]")[1]
    }, raw_rarank_data))


def mean_attention_heads(attentions_layers_tuple):
    return torch.stack([torch.mean(layer, dim=1) for layer in attentions_layers_tuple])


def get_current_dialog(selected_dialog):
    diag_turns = selected_dialog["dialog"]["turns"]
    rerank_dialog_examples = split_utterance_history(selected_dialog["to_rerank"])

    if "gpt_references" in selected_dialog:
        for psg_id, gpt_ref in selected_dialog["gpt_references"].items():
            psg_id = int(psg_id)
            rerank_dialog_examples[psg_id]["gpt_references"] = gpt_ref

    utterance_history, passage = [(d["utterance_history"], d["passage"])
                                  for d in rerank_dialog_examples
                                  if d["label"] == 1][0]
    last_user_utterance = utterance_history.split("agent: ")[0]
    last_user_utterance_id = [t['turn_id'] for t in diag_turns if t["utterance"] == last_user_utterance][0]
    grounded_agent_utterance_id = last_user_utterance_id  # Ids stats from 1 and agent utterance is the next after user
    grounded_agent_utterance = diag_turns[grounded_agent_utterance_id]
    nr_show_utterances = grounded_agent_utterance_id + 1
    return diag_turns, grounded_agent_utterance, nr_show_utterances, rerank_dialog_examples


def cross_encoder_inference(rerank_dialog_examples, tokenizer, cross_encoder, max_to_rerank=32):
    pre_examples = preprocess_examples(rerank_dialog_examples, tokenizer, 512)
    batch = utils.transform_batch(pre_examples, max_to_rerank, device=device)

    cross_encoder.save_att_gradients = True
    cross_encoder.save_output_gradients = True
    cross_encoder.save_outputs = True

    logits = []
    attention_maps = []
    grad_sam_scores = []
    att_cat_scores = []
    for in_ids, att_mask, tt_ids in zip(batch['in_ids'], batch['att_mask'], batch['tt_ids']):
        logit = cross_encoder(
            input_ids=in_ids[None, :],
            attention_mask=att_mask[None, :],
            token_type_ids=tt_ids[None, :],
        )
        logits.append(logit)
        logit.backward()

        with torch.no_grad():
            # Get gradients, outputs and attention weights
            att_grads = torch.stack(cross_encoder.get_att_gradients()).squeeze()
            att_weights_tuple = cross_encoder.get_attention_weights()
            outputs = cross_encoder.saved_outputs
            out_grads = cross_encoder.saved_output_gradients

            # Stack attention weights
            atts = torch.stack([layer.squeeze() for layer in att_weights_tuple])
            attention_maps.append(atts)

            grad_sam_score = grad_sam(atts, att_grads)
            grad_sam_scores.append(grad_sam_score)

            att_cat_score = att_cat(outputs, out_grads, atts)
            att_cat_scores.append(att_cat_score)

    logits = torch.tensor(logits)

    # Compute attention rollout
    # torch.Size([B, 24, 16, 512, 512])  print(att_weights.shape)
    att_weights_norm_heads = mean_attention_heads(attention_maps)
    # torch.Size([B, 24, 512, 512]) print(att_weights_norm_heads.shape)
    batched_rollouts = attention_rollout(torch.permute(att_weights_norm_heads, (1, 0, 2, 3)))
    # torch.Size([B, 512, 512]) print(batched_rollouts.shape)
    batched_rollouts_cls = batched_rollouts[:, 0, :]
    # torch.Size([32, 512]) print(batched_rollouts[:, 0, :].shape)

    # Sorting wrt predictions
    sorted_indexes = torch.argsort(logits, descending=True).cpu()

    sorted_logits = [logits[i] for i in sorted_indexes]
    reranked_examples = [rerank_dialog_examples[i] for i in sorted_indexes]
    reranked_att_weights_cls = [attention_maps[i][:, :, 0, :].cpu() for i in sorted_indexes]
    reranked_rollouts = [batched_rollouts_cls[i].cpu() for i in sorted_indexes]
    reranked_grad_sam_scores = [grad_sam_scores[i].cpu() for i in sorted_indexes]
    reranked_att_cat_scores = [att_cat_scores[i].cpu() for i in sorted_indexes]
    torch.cuda.empty_cache()
    return {
        "reranked_examples": reranked_examples,
        "sorted_predictions": sorted_logits,
        "reranked_rollouts": reranked_rollouts,
        "att_weights_cls": reranked_att_weights_cls,  # (B, L, H, S)
        "grad_sam_scores": reranked_grad_sam_scores,
        "att_cat_scores": reranked_att_cat_scores,
        "sorted_indexes": sorted_indexes
    }


def generate_metadata():
    diag_examples = []
    for dialog_idx in tqdm(range(200), desc="Getting metadata information"):
        try:
            diag_example = torch.load(f"data/examples/inference/{dialog_idx}_dialogue_reranking_inference.pt")
        except FileNotFoundError:
            # A few dialogues are skipped during inference
            continue

        diag_examples.append(
            {
                "diag_id": diag_example["diag_id"],
                "gt_example_prediction": diag_example["gt_example_prediction"].tolist(),
                "gt_rank": diag_example["gt_rank"],
                "nr_turns": diag_example["nr_show_utterances"],
            }
        )

    diag_examples = sorted(diag_examples, key=lambda x: x['gt_example_prediction'])
    json.dump(diag_examples, open("data/examples/inference/diag_examples_inference_out.json", "w"))


def generate_offline_data(from_id=0, to_id=200, refresh_metadata=True):
    # Code here is used to generate offline data for the visualization
    data_dialogues = get_data()
    cross_encoder, tokenizer = init_model()

    for dialog_idx in tqdm(range(from_id, to_id), desc="Inferecing dialogues"):
        selected_dialogue = data_dialogues[dialog_idx]
        diag_turns, grounded_agent_utterance, nr_show_utterances, rerank_dialog_examples \
            = get_current_dialog(selected_dialogue)

        inf_out = cross_encoder_inference(rerank_dialog_examples, tokenizer, cross_encoder)

        try:
            gt_rank = [i
                       for i, reranked_example in enumerate(inf_out['reranked_examples'])
                       if reranked_example['label'] == 1][0]
        except IndexError:
            print(f"Skipping id {dialog_idx}")
            continue
        gt_example_prediction = inf_out['sorted_predictions'][gt_rank]

        diag_example = {
            "diag_id": dialog_idx,
            "diag_turns": diag_turns,
            "grounded_agent_utterance": grounded_agent_utterance,
            "nr_show_utterances": nr_show_utterances,
            "rerank_dialog_examples": rerank_dialog_examples,
            "gt_rank": gt_rank,
            "gt_example_prediction": gt_example_prediction,
            "inf_out": inf_out,
        }

        torch.save(diag_example, f"data/examples/inference/{dialog_idx}_dialogue_reranking_inference.pt")

    if refresh_metadata:
        generate_metadata()


def split_to_spans(sentences):
    spans = []
    spans_split_by = ["//", ","]
    for sentence in sentences:
        if "[SEP]" in sentence:
            sep_split1, sep_split2 = sentence.split("[SEP]")
            if "//" in sep_split2:
                before, after = sep_split2.split("//", 1)
                spans.extend([sep_split1, "[SEP]", before + "//", after])
            else:
                spans.extend([sep_split1, "[SEP]", sep_split2])
            continue

        added_split = False
        for split_by in spans_split_by:
            if split_by in sentence:
                before, after = sentence.split(split_by, 1)
                spans.extend([before + split_by, after])
                added_split = True
                break

        if not added_split:
            spans.append(sentence)

    return spans


def score_to_span_scores(grad_sam_scores, spans, tokenizer):
    nr_tokens_all = 0
    new_grad_sam_scores = []
    for span in spans:
        nr_tokens_span = len(split_to_tokens(span, tokenizer))
        if nr_tokens_span == 0:
            continue
        nr_tokens_all += nr_tokens_span

        grad_sam_score_spans = grad_sam_scores[nr_tokens_all: nr_tokens_all + nr_tokens_span]
        # new_grad_sam_score = torch.mean(grad_sam_score_spans)
        t = nr_tokens_span
        N = 13
        k = 4
        new_grad_sam_score = torch.sum(grad_sam_score_spans) / ((1 + N) * (t + k) / t)
        new_grad_sam_scores.append(torch.ones(nr_tokens_span) * new_grad_sam_score)
    return torch.cat(new_grad_sam_scores)


class InferenceDataProvider:
    def __init__(self, cross_encoder=None, tokenizer=None):
        self.cross_encoder = cross_encoder
        self.tokenizer = tokenizer
        self.meta_data = json.load(open("data/examples/inference/diag_examples_inference_out.json"))
        self.data = None  # Data will be loaded online in mode
        self.last_loaded_example = None  # Last loaded example for offline mode
        self.last_rerank_dialog_examples = None  # used to remember dialog for online inference

        self.mode = None
        self.set_mode("offline")

    def set_mode(self, new_mode):
        if self.mode == new_mode:
            return

        if new_mode == "online":
            self.data = get_data()
        elif new_mode == "offline":
            pass
        self.mode = new_mode

    def get_dialog_inference_out(self, dialog_id, max_to_rerank=32):
        assert self.mode in ["online", "offline"]

        if self.mode == "offline":
            self.get_dialog_out(dialog_id)

            inf_out = self.last_loaded_example["inf_out"]
            for inf_key in inf_out.keys():
                inf_out[inf_key] = inf_out[inf_key][:max_to_rerank]

            return inf_out
        elif self.mode == "online":
            assert self.cross_encoder is not None
            self.get_dialog_out(dialog_id)
            return cross_encoder_inference(self.last_rerank_dialog_examples,
                                           max_to_rerank,
                                           self.tokenizer,
                                           self.cross_encoder)

    def get_dialog_out(self, dialog_id):
        assert self.mode in ["online", "offline"]

        if self.mode == "offline":
            self.last_loaded_example = self.load_example(dialog_id)
            diag_turns = self.last_loaded_example["diag_turns"]
            grounded_agent_utterance = self.last_loaded_example["grounded_agent_utterance"]
            nr_show_utterances = self.last_loaded_example["nr_show_utterances"]
            rerank_dialog_examples = self.last_loaded_example["rerank_dialog_examples"]
            return diag_turns, grounded_agent_utterance, nr_show_utterances, rerank_dialog_examples
        elif self.mode == "online":
            diag_turns, grounded_agent_utterance, nr_show_utterances, rerank_dialog_examples \
                = get_current_dialog(self.data[dialog_id])
            self.last_rerank_dialog_examples = rerank_dialog_examples
            return diag_turns, grounded_agent_utterance, nr_show_utterances, rerank_dialog_examples

    @staticmethod
    def load_example(dialog_id):
        return torch.load(f"data/examples/inference/{dialog_id}_dialogue_reranking_inference.pt")

    def get_nr_dialogs(self):
        return len(self.meta_data)

    def get_inf_sorted_dialog_ids(self):
        data = [md["diag_id"] for md in self.meta_data]
        data.reverse()
        return data

    def get_valid_dialog_ids(self):
        return sorted(self.get_inf_sorted_dialog_ids())
