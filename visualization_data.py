import json

import nltk
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import utils
from interpretability import attention_rollout, grad_sam
from md2d_dataset import preprocess_examples
from train_ce import CrossEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXAMPLE_VALIDATION_DATA = "data/examples/200_dialogues_reranking.json"
model_name = "naver/trecdl22-crossencoder-debertav3"


def init_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cross_encoder = CrossEncoder(model_name)
    cross_encoder.save_attention_weights = True
    cross_encoder.bert_model.config.output_attentions = True
    utils.load_model(cross_encoder, "CE_lr0.00010485388296357131_bs16.pt")
    cross_encoder.to(device)
    cross_encoder.eval()
    return cross_encoder, tokenizer


cross_encoder, tokenizer = init_model()


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


def get_current_dialog(loaded_data_dialogues, current_id):
    selected_dialog = loaded_data_dialogues[current_id]
    diag_turns = selected_dialog["dialog"]["turns"]
    rerank_dialog_examples = split_utterance_history(selected_dialog["to_rerank"])
    utterance_history, passage = [(d["utterance_history"], d["passage"])
                                  for d in rerank_dialog_examples
                                  if d["label"] == 1][0]
    last_user_utterance = utterance_history.split("agent: ")[0]
    last_user_utterance_id = [t['turn_id'] for t in diag_turns if t["utterance"] == last_user_utterance][0]
    grounded_agent_utterance_id = last_user_utterance_id  # Ids stats from 1 and agent utterance is the next after user
    grounded_agent_utterance = diag_turns[grounded_agent_utterance_id]
    nr_show_utterances = grounded_agent_utterance_id + 1
    return diag_turns, grounded_agent_utterance, nr_show_utterances, rerank_dialog_examples


def cross_encoder_inference(rerank_dialog_examples):
    max_to_rerank = 32
    pre_examples = preprocess_examples(rerank_dialog_examples, tokenizer, 512)
    batch = utils.transform_batch(pre_examples, max_to_rerank, device=device)

    cross_encoder.save_gradients = True
    logits = []
    attention_maps = []
    grad_sam_scores = []
    for in_ids, att_mask, tt_ids in zip(batch['in_ids'], batch['att_mask'], batch['tt_ids']):
        logit = cross_encoder(
            input_ids=in_ids[None, :],
            attention_mask=att_mask[None, :],
            token_type_ids=tt_ids[None, :],
        )
        logits.append(logit)
        logit.backward()

        with torch.no_grad():
            # Get gradients and attention weights
            grads = torch.stack(cross_encoder.get_gradients()).squeeze()
            att_weights_tuple = cross_encoder.get_attention_weights()

            # Stack attention weights
            atts = torch.stack([layer.squeeze() for layer in att_weights_tuple])
            attention_maps.append(atts)

            grad_sam_score = grad_sam(atts, grads)
            grad_sam_scores.append(grad_sam_score)

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
    torch.cuda.empty_cache()
    return {
        "reranked_examples": reranked_examples,
        "sorted_predictions": sorted_logits,
        "reranked_rollouts": reranked_rollouts,
        "att_weights_cls": reranked_att_weights_cls,  # (B, L, H, S)
        "grad_sam_scores": reranked_grad_sam_scores
    }


def generate_offline_data():
    # Code here is used to generate offline data for the visualization
    data_dialogues = get_data()

    diag_examples = []
    for dialog_idx in tqdm(range(200), desc="Dialogs"):
        diag_turns, grounded_agent_utterance, nr_show_utterances, rerank_dialog_examples \
            = get_current_dialog(data_dialogues, dialog_idx)

        inf_out = cross_encoder_inference(rerank_dialog_examples)

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

        diag_examples.append(
            {
                "diag_id": dialog_idx,
                "gt_example_prediction": gt_example_prediction.tolist(),
                "gt_rank": gt_rank,
                "nr_turns": nr_show_utterances
            }
        )

        torch.save(diag_example, f"data/examples/inference/{dialog_idx}_dialogue_reranking_inference.pt")

    diag_examples = sorted(diag_examples, key=lambda x: x['gt_example_prediction'])
    json.dump(diag_examples, open("data/examples/inference/diag_examples_inference_out.json", "w"))


def split_to_tokens(text):
    return tokenizer.batch_decode(tokenizer(text)["input_ids"])[1:][:-1]


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


def score_to_span_scores(grad_sam_scores, spans):
    nr_tokens_all = 0
    new_grad_sam_scores = []
    for span in spans:
        nr_tokens_span = len(split_to_tokens(span))
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


if __name__ == "__main__":
    cross_encoder, tokenizer = init_model()

    for dialog_id in tqdm(range(200), desc="Dialogs"):
        try:
            last_loaded_example = torch.load(
                f"data/examples/inference/{dialog_id}_dialogue_reranking_inference.pt")
        except FileNotFoundError:
            continue

        span_grad_sam_scores = []
        for i, example in enumerate(last_loaded_example["inf_out"]["reranked_examples"]):
            sentences = nltk.sent_tokenize(example["x"])
            example["spans"] = split_to_spans(sentences)

            grad_sam_scores = last_loaded_example["inf_out"]["grad_sam_scores"][i]
            grad_sam_scores_spans = score_to_span_scores(grad_sam_scores, example["spans"])
            span_grad_sam_scores.append(grad_sam_scores_spans)

        last_loaded_example["inf_out"]["grad_sam_scores_spans_K_N"] = span_grad_sam_scores

        torch.save(last_loaded_example, f"data/examples/inference/{dialog_id}_dialogue_reranking_inference.pt")


class InferenceDataProvider:
    def __init__(self, cross_encoder, tokenizer):
        self.mode = None
        self.cross_encoder = cross_encoder
        self.tokenizer = tokenizer
        self.meta_data = json.load(open("data/examples/inference/diag_examples_inference_out.json"))
        self.data = None  # Data will be loaded online in mode
        self.last_loaded_example = None  # Last loaded example for offline mode
        self.last_rerank_dialog_examples = None  # used to remember dialog for online inference

    def set_mode(self, new_mode):
        if self.mode == new_mode:
            return

        if new_mode == "online":
            self.data = get_data()
        elif new_mode == "offline":
            pass
        self.mode = new_mode

    def get_dialog_inference_out(self, dialog_id):
        assert self.mode in ["online", "offline"]

        if self.mode == "offline":
            return self.last_loaded_example["inf_out"]
        elif self.mode == "online":
            return cross_encoder_inference(self.last_rerank_dialog_examples)

    def get_dialog_out(self, dialog_id):
        assert self.mode in ["online", "offline"]

        if self.mode == "offline":
            self.last_loaded_example = torch.load(
                f"data/examples/inference/{dialog_id}_dialogue_reranking_inference.pt")
            diag_turns = self.last_loaded_example["diag_turns"]
            grounded_agent_utterance = self.last_loaded_example["grounded_agent_utterance"]
            nr_show_utterances = self.last_loaded_example["nr_show_utterances"]
            rerank_dialog_examples = self.last_loaded_example["rerank_dialog_examples"]
            return diag_turns, grounded_agent_utterance, nr_show_utterances, rerank_dialog_examples
        elif self.mode == "online":
            diag_turns, grounded_agent_utterance, nr_show_utterances, rerank_dialog_examples \
                = get_current_dialog(self.data, dialog_id)
            self.last_rerank_dialog_examples = rerank_dialog_examples
            return diag_turns, grounded_agent_utterance, nr_show_utterances, rerank_dialog_examples

    def get_nr_dialogs(self):
        return len(self.meta_data)

    def get_sorted_dialogs(self):
        data = [md["diag_id"] for md in self.meta_data]
        data.reverse()
        return data
