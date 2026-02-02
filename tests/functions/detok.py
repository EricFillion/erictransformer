import csv
import json
import os

from erictransformer import EricTransformer
from erictransformer.eric_tasks.eric_generation import GENTokArgs


def t_detok_csv(eric_model: EricTransformer, input_data: str, out_dir: str):
    save_path = os.path.join(out_dir, "csv_decode")

    eric_model.tok(input_data, save_path)

    tok_data = eric_model.tok_dir_to_dataset(save_path)

    lines = []
    with open(input_data, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # we ignore capitalization for simplicity.
            lines.append(row["text"].lower())

    for case in tok_data["input_ids"]:
        detok_case = eric_model.tokenizer.decode(
            case, skip_special_tokens=True
        ).lower()

        print(detok_case)
        print(lines)
        assert detok_case in lines


def t_detok_text(
    eric_model: EricTransformer,
    input_data: str,
    out_dir: str,
    task_type: str = "gen",
):
    save_path = os.path.join(out_dir, "text_decode")
    eric_model.tok(input_data, save_path)

    tok_data = eric_model.tok_dir_to_dataset(save_path)
    with open(input_data, newline="") as text_file:
        file_contents = text_file.read()
        if task_type == "wp":
            file_contents = file_contents.lower().replace("\n", " ")
        else:
            file_contents = file_contents + "\n"

    for case in tok_data["input_ids"]:
        detok_case = eric_model.tokenizer.decode(case, skip_special_tokens=True)
        if task_type == "wp":
            detok_case = detok_case.lower().replace("\n", " ")

        assert detok_case in file_contents


def t_detok_jsonl(
    eric_model: EricTransformer,
    input_data: str,
    out_dir: str,
    text_field: str = "text",
):
    save_path = os.path.join(out_dir, "jsonl_decode")
    eric_model.tok(input_data, save_path)
    tok_data = eric_model.tok_dir_to_dataset(save_path)

    with open(input_data, "r", encoding="utf-8") as f:
        # we ignore capitalization for simplicity
        file_contents = [json.loads(line)[text_field].lower() for line in f]

    for case, original_text in zip(tok_data["input_ids"], file_contents):
        detok_case = eric_model.tokenizer.decode(
            case, skip_special_tokens=True
        ).lower()
        assert detok_case == original_text


def t_detok_len(
    eric_model: EricTransformer, input_data: str, out_dir: str, args: GENTokArgs
):
    MAX_LEN = 2

    args.max_len = MAX_LEN

    save_path = os.path.join(out_dir, "text_len")

    eric_model.tok(input_data, save_path, args=args)

    tok_data = eric_model.tok_dir_to_dataset(save_path)

    for case in tok_data["input_ids"]:
        assert len(case) == MAX_LEN
