import argparse
import json
import os
from datetime import datetime

from jsonlines import jsonlines
from openai import OpenAI
from tqdm import tqdm

from visualization_data import init_model, InferenceDataProvider, get_data, get_current_dialog, EXAMPLE_VALIDATION_DATA, \
    generate_offline_data, generate_metadata


class OpenAIGenerator:
    def __init__(self):
        self.client = OpenAI()
        self.temperature = 0.2
        self.max_tokens = 1024
        self.top_p = 1
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.system_message = "You are helpful linguistic specialist eager to complete given task."

    def create_message(self, prompt):
        return [
            {
                "role": "system",
                "content": self.system_message
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

    def create_api_call_dict(self, message, from_prompt=True):
        """
        Create API call for OpenAI API
        :param message: Prompt for the API call or message generated with create_message()
        :param from_prompt: Use to transform user prompt to conversation format
        :return:
        """
        if from_prompt:
            message = self.create_message(message)

        return dict(
            model="gpt-4o-2024-08-06",
            messages=message,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            response_format={
                "type": "json_object"
            }
        )

    def __call__(self, message):
        api_dict = self.create_api_call_dict(message)
        generation_result = self.client.chat.completions.create(**api_dict)
        generated_answer = generation_result.choices[0].message.content
        return generated_answer


def turns_for_gpt(turns):
    gpt_diag_turns = []
    # remove last turn as it is the one generated from the agent
    for turn in turns[:-1]:
        gpt_diag_turns.append({
            "role": turn["role"],
            "turn_id": turn["turn_id"],
            "utterance": turn["utterance"]
        })
    return gpt_diag_turns


def task_from_prompt(custom_id, prompt):
    return {
        "custom_id": f"{custom_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": prompt
    }


def messages_for_passages(turns, diag_passages, openai_api, nr_passages=16):
    turns_str = json.dumps(turns, indent=4)
    messages = []
    for psg_id in tqdm(range(nr_passages), desc="Passages"):
        psg_text = diag_passages[psg_id]["passage"]
        prompt_str = (
            "You will be presented with a dialogue in json format followed by partially relevant text. Your task is "
            "to select spans containing answer to the last question based on entire dialogue history.  You may select "
            "multiple spans if needed, but ensure that the selected sections do not overlap. Try to not select entire "
            "sentences, but only fine-grained spans."
            "Return json_object with key 'spans' and list of selected spans as value. "
            "\n"
            "Dialogue history:"
            f"{turns_str}\n"
            "\n"
            "Text:"
            f"{psg_text}"
        )
        api_message = openai_api.create_api_call_dict(prompt_str)
        messages.append(api_message)
    return messages


def init_writer(dialogue_id_from, dialogue_id_to):
    time_str = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    jsonl_filename = f"openAI_batches/{time_str}_batch-{dialogue_id_from}-{dialogue_id_to}.jsonl"
    return jsonl_filename, jsonlines.Writer(open(jsonl_filename, "a"))


def create_batch_file(dialogue_id_from, dialogue_id_to):
    """
    Create batch file for a range of dialogues in a jsonl format, which is suitable for OpenAI API.
    https://platform.openai.com/docs/guides/batch/getting-started
    :param dialogue_id_from: start index of the batch
    :param dialogue_id_to: end index of the batch
    :return: None, but jsonl file is created
    """
    cross_encoder, tokenizer = init_model()
    offline_inference = InferenceDataProvider(cross_encoder, tokenizer)
    openai_api = OpenAIGenerator()
    jsonl_filename, task_writer = init_writer(dialogue_id_from, dialogue_id_to)
    data = get_data()

    batch_psg_ids = offline_inference.get_valid_dialog_ids()[dialogue_id_from:dialogue_id_to]
    for dialogue_id in tqdm(batch_psg_ids, desc="Dialogs"):

        # Get next dialogue data
        last_loaded_example = data[dialogue_id]
        print(f"Generating for dialogue id {dialogue_id}")
        diag_turns, grounded_agent_utterance, nr_show_utterances, diag_passages = \
            get_current_dialog(last_loaded_example)
        turns = turns_for_gpt(diag_turns[:nr_show_utterances])

        # Rerank passages based on inference sorted indexes
        inf_out = offline_inference.get_dialog_inference_out(dialogue_id)
        diag_passages = [diag_passages[i] for i in inf_out["sorted_indexes"]]
        messages = messages_for_passages(turns, diag_passages, openai_api)

        # Create sub-batch of passages for the dialogue
        tasks = []
        for id_m, message in enumerate(messages):
            diag_psg_id = f"{dialogue_id}_{id_m}"
            task = task_from_prompt(diag_psg_id, message)
            tasks.append(task)

        # Append tasks to the file
        task_writer.write_all(tasks)

    print(f"Batch file saved to {jsonl_filename}")


def create_batch_job(batch_filename):
    client = OpenAI()
    batch_input_file = client.files.create(
        file=open(batch_filename, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"job on filename {batch_filename}",
            "for_filename": batch_filename
        }
    )

    print(f"Created batch ({batch.id}):")
    print(batch)


def get_output_batch(batch_id):
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    if batch.status == "completed":
        print("Batch is completed")
        result_file_id = batch.output_file_id
        result = client.files.content(result_file_id).content
        batch_filename = batch.metadata["for_filename"].strip('.jsonl')
        result_file_name = f"openAI_batches/{batch_filename}_output.jsonl"
        with open(result_file_name, 'wb') as file:
            file.write(result)
        print(f"Output file saved to {result_file_name}")
    else:
        print("\n\n\n")
        print(f"All batches:")
        for b in client.batches.list():
            print(b, "\n\n")

        print(f"Batch is not completed yet - status is {batch.status}")
        print(batch)


def message_from_output(res):
    message_json = res['response']['body']['choices'][0]['message']['content']
    try:
        message = json.loads(message_json)
    except json.JSONDecodeError:
        # Remove code block markdown (generated response is not json_obj but text)
        message = message_json.strip("```").strip("json\n")
        message = json.loads(message)
    return message


def messages_from_file(batch_filename):
    messages = []
    with jsonlines.open(batch_filename) as reader:
        for res in reader:
            message = message_from_output(res)
            messages.append({
                "spans": message["spans"],
                "id": res["custom_id"]
            })
    return messages


def process_output(batch_filename, refresh_metadata=True):
    cross_encoder, tokenizer = init_model()
    messages = messages_from_file(batch_filename)
    data = get_data()
    offline_inference = InferenceDataProvider(cross_encoder, tokenizer)

    for message in messages:
        dialogue_id, sorted_passage_id = message["id"].split("_")
        dialogue_id = int(dialogue_id)

        # Get reranked passage indexes because passage_id from "custom_id" is from reranked list
        sorted_psg_indexes = offline_inference.get_dialog_inference_out(dialogue_id)["sorted_indexes"]
        real_psg_id = sorted_psg_indexes[int(sorted_passage_id)]

        # Result not generated for all passages therefore gpt_references is dict
        # where key is real_psg_id and value is list of gpt selected spans
        real_psg_id = str(int(real_psg_id))
        if "gpt_references" not in data[dialogue_id]:
            data[dialogue_id]["gpt_references"] = {}

        messages = [
            {
                "ref_span": span,
            }
            for span in message["spans"]
        ]
        data[dialogue_id]["gpt_references"][real_psg_id] = messages

    json.dump(data, open(EXAMPLE_VALIDATION_DATA, "w"), indent=4)
    from_id, to_id = batch_filename.split("_batch-")[1].strip("_output.jsonl").split("-")
    generate_offline_data(int(from_id), int(to_id), refresh_metadata=refresh_metadata)


def asdf():
    data = get_data()
    for diag_id in range(200):
        if "gpt_references" in data[diag_id]:
            if isinstance(data[diag_id]["gpt_references"], list):
                del data[diag_id]["gpt_references"]
    json.dump(data, open(EXAMPLE_VALIDATION_DATA, "w"), indent=4)


if __name__ == "__main__":
    argparse.ArgumentParser(description='Generate explanations for MD2D dataset')
    parser = argparse.ArgumentParser()

    # Generate explanations for a batch of samples
    parser.add_argument('--generate-batch',
                        action='store_true', help='Generate explanations for a batch of samples')
    parser.add_argument("--from-sample", type=int, default=0, help="Start index of the batch")
    parser.add_argument("--to-sample", type=int, default=1, help="End index of the batch")

    # Create batch job OpenAI API
    parser.add_argument("--create-batch-job",
                        action='store_true', help='Upload batch to OpenAI API')
    parser.add_argument("--batch-filename", type=str,
                        help="Path to the batch file")

    # Check batch status and get output 
    parser.add_argument("--get-batch-output",
                        action='store_true', help='Check batch status and get output')
    parser.add_argument("--batch-id", type=str, help="Batch ID")

    # Process downloaded batch output
    parser.add_argument("--process-output",
                        action='store_true', help='Process output from OpenAI API')
    parser.add_argument("--batch-out-filename", type=str,
                        help="Path to the file with OpenAI API output")

    # Process all outputs (similar as previous but for all)
    parser.add_argument("--process-all",
                        action='store_true', help='Process all outputs from OpenAI API')

    args = parser.parse_args()

    if args.generate_batch:
        create_batch_file(args.from_sample, args.to_sample)
    elif args.create_batch_job:
        create_batch_job(args.batch_filename)
    elif args.get_batch_output:
        get_output_batch(args.batch_id)
    elif args.process_output:
        process_output(args.batch_out_filename)
    elif args.process_all:
        base_dir = "openAI_batches"
        for filename in os.listdir(base_dir):
            if filename.endswith("_output.jsonl"):
                out_filename = f"{base_dir}/{filename}"
                print(f"Processing {out_filename}")
                process_output(out_filename, refresh_metadata=False)
        generate_metadata()

    else:
        print("No action specified")
        parser.print_help()
