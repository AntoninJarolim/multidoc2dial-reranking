import json

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer


def split_on_SEP(obj):
    return obj.split("[SEP]")


def load_example(obj, all_dialogs, all_passages):
    dialog, passage = split_on_SEP(obj["x"])
    all_dialogs.add(dialog)
    all_passages.add(passage)


def load_data(split, all_dialogs, all_passages, validation=False):
    with open(f"data/DPR_pairs/DPR_pairs_{split}.jsonl", "r") as f:
        for line in f:
            decoded_line = json.loads(line)

            if not validation:
                load_example(decoded_line, all_dialogs, all_passages)
            else:
                for query in decoded_line:
                    load_example(query, all_dialogs, all_passages)


def load_dataset():
    all_dialogs = set()
    all_passages = set()
    for split in ["train"]:  # , "validation", "test"]:
        load_data(split, all_dialogs, all_passages)

    for split in ["validation", "test"]:
        load_data(split, all_dialogs, all_passages, validation=True)

    print(f"Number of dialogs: {len(all_dialogs)}")
    print(f"Number of passages: {len(all_passages)}")
    return list(all_dialogs), list(all_passages)


# get dataset
all_dialogs, all_passages = load_dataset()
tokenizer = AutoTokenizer.from_pretrained("naver/trecdl22-crossencoder-debertav3")
vocab = tokenizer.get_vocab()

## Fit tf-idf
vectorizer_dialogs = TfidfVectorizer(tokenizer=tokenizer.tokenize, vocabulary=vocab)
fitted_dialogs = vectorizer_dialogs.fit(all_dialogs)
terms_dialogs = vectorizer_dialogs.get_feature_names_out()

vectorizer_docs = TfidfVectorizer(tokenizer=tokenizer.tokenize, vocabulary=vocab)
fitted_docs = vectorizer_docs.fit(all_passages)
terms_docs = vectorizer_docs.get_feature_names_out()

terms = terms_docs


def token_overlap_score(doc_term_idxs, scores, debug=True):
    term_scores = {}
    for idx in doc_term_idxs:  # non_zero_indices[1] gives column indices
        term_scores[terms[idx]] = scores[idx]

        # Sort by insertion
    term_scores = {k: v for k, v in sorted(term_scores.items(), key=lambda item: -item[1])}

    if debug:
        for k, v in term_scores.items():
            print(f"{term_scores[score]}: {score}")

    return term_scores


def score_passage_overlap(diag_sep_passage):
    diag_str, doc_str = split_on_SEP(diag_sep_passage)

    diag_tranf = fitted_dialogs.transform([diag_str])[0]
    doc_tranf = fitted_docs.transform([doc_str])[0]

    scores = diag_tranf.toarray()[0] * doc_tranf.toarray()[0]
    overlap_idxs = scores.nonzero()[0]
    scores_nonzero = scores[overlap_idxs]

    overlap_dict = token_overlap_score(overlap_idxs, scores, debug=False)

    tokenized_doc = tokenizer.tokenize(doc_str)
    scores_passage = torch.rand(len(tokenized_doc)) * 1e-6
    for i, t in enumerate(tokenized_doc):
        scores_passage[i] = overlap_dict.get(t, 0)

    return scores_passage


x = 'My insurance ended so what should i do[SEP]Respond to DMV insurance letters and orders //   I received a Letter that states my insurance lapsed. What can I do?   The letter means your insurance company notified the DMV that your insurance coverage ended, and that no other company notified the DMV about new coverage.  If you have insurance ,  follow the instructions in the letter and contact your company or agent about the problem.  Ask your company to file a notice of coverage with the DMV electronically.  If you don t have insurance ,  you need to surrender your vehicle registration and plates to the DMV immediately.'

score_passage_overlap(x)
