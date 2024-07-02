import json

dialog_path = 'data/data/multidoc2dial/multidoc2dial_dial_train.json'
passage_path = 'data/data/multidoc2dial/multidoc2dial_doc.json'

oa_index = {}

passage_data = json.load(open(passage_path, "r"))


def get_span(domain, doc_id, span_id):
    passage = passage_data['doc_data'][domain][doc_id]
    spans = passage['spans']
    start_span = spans[span_id]['start_sp']
    end_span = spans[span_id]['end_sp']
    return passage['doc_text'][start_span:end_span]


dialog_data = json.load(open(dialog_path, "r"))
for domain, values_domain in dialog_data['dial_data'].items():
    for diag_index, dialog in enumerate(values_domain):
        for turn_id, turn in enumerate(dialog['turns']):
            for ref_id, reference in enumerate(turn['references']):
                doc_id = reference['doc_id']
                span_id = reference['id_sp']
                span = get_span(domain, doc_id, span_id)
                dialog_data['dial_data'][domain][diag_index]['turns'][turn_id]['references'][ref_id]['ref_span'] = span

out_dialog_path = dialog_path.strip(".json") + "_refs.json"
json.dump(dialog_data, open(out_dialog_path, 'w'), indent=4)
