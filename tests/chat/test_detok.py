import os

from erictransformer import CHATTokArgs
from tests.chat import CHAT_DATA_JSONL, CHAT_out_dir_DATA, eric_chat


def test_detok_length():
    MAX_LEN = 2

    save_path = os.path.join(CHAT_out_dir_DATA, "text_len")

    eric_chat.tok(CHAT_DATA_JSONL, save_path, args=CHATTokArgs(max_len=MAX_LEN))

    tok_data = eric_chat.tok_dir_to_dataset(save_path)

    for case in tok_data["input_ids"]:
        r = eric_chat.tokenizer.decode(case)
        assert len(case) == MAX_LEN
