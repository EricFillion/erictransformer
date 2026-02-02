import csv
import json
import os

from erictransformer import TTTokArgs
from tests.tt import TT_DATA_JSONL, TT_out_dir_DATA, eric_tt



def test_detok_jsonl():
    save_path = os.path.join(TT_out_dir_DATA, "jsonl_decode")
    eric_tt.tok(TT_DATA_JSONL, save_path)
    tok_data = eric_tt.tok_dir_to_dataset(save_path)
    inputs = []
    targets = []

    with open(TT_DATA_JSONL, "r", encoding="utf-8") as f:
        # we ignore capitalization for simplicity
        for line in f:
            if not line.strip():
                break
            line_dict = json.loads(line)

            inputs.append(line_dict["input"].lower())
            targets.append(line_dict["target"].lower())

    for case, original_text in zip(tok_data["input_ids"], inputs):
        detok_case = eric_tt.tokenizer.decode(case, skip_special_tokens=True).lower()
        assert detok_case == original_text

    for case, original_text in zip(tok_data["labels"], targets):
        detok_case = eric_tt.tokenizer.decode(
            [t for t in case if t != -100], skip_special_tokens=True
        ).lower()
        assert detok_case == original_text


def test_detok_length():
    MAX_INPUT_LEN = 2
    MAX_OUTPUT_LEN = 3

    args = TTTokArgs(max_in_len=MAX_INPUT_LEN, max_out_len=MAX_OUTPUT_LEN)

    save_path = os.path.join(TT_out_dir_DATA, "text_len")

    eric_tt.tok(TT_DATA_JSONL, save_path, args=args)

    tok_data = eric_tt.tok_dir_to_dataset(save_path)

    for case in tok_data["input_ids"]:
        assert len(case) == MAX_INPUT_LEN

    for case in tok_data["labels"]:
        assert len(case) == MAX_OUTPUT_LEN
