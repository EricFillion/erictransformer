from erictransformer import CHATTokArgs
from erictransformer.eric_tasks.tok.tok_functions import get_max_in_len
from tests.chat import CHAT_DATA_JSONL, CHAT_out_dir_DATA, eric_chat
from tests.functions.tok_length import t_tok_length


def test_pad():
    target_length = 256

    args = CHATTokArgs(max_len=target_length)

    t_tok_length(
        eric_model=eric_chat,
        input_data_list=[CHAT_DATA_JSONL],
        out_dir=CHAT_out_dir_DATA,
        args=args,
        target_length=target_length,
    )


def test_pad_max():
    target_length = get_max_in_len(
        max_len=-1, tokenizer=eric_chat.tokenizer
    )
    args = CHATTokArgs()
    t_tok_length(
        eric_model=eric_chat,
        input_data_list=[CHAT_DATA_JSONL],
        out_dir=CHAT_out_dir_DATA,
        args=args,
        target_length=target_length,
    )


def test_trunc():
    target_length = 4
    args = CHATTokArgs(max_len=target_length)

    t_tok_length(
        eric_model=eric_chat,
        input_data_list=[CHAT_DATA_JSONL],
        out_dir=CHAT_out_dir_DATA,
        args=args,
        target_length=target_length,
    )
