from erictransformer import TTTokArgs
from erictransformer.eric_tasks.tok.tok_functions import get_max_in_len
from tests.functions.tok_length import t_tok_length
from tests.tt import TT_DATA_JSONL, TT_out_dir_DATA, eric_tt


def test_pad():
    target_length = 256

    args = TTTokArgs(max_in_len=target_length, max_out_len=target_length)

    t_tok_length(
        eric_model=eric_tt,
        input_data_list=[TT_DATA_JSONL],
        out_dir=TT_out_dir_DATA,
        args=args,
        target_length=target_length,
    )


def test_pad_max():
    target_length = get_max_in_len(max_len=-1, tokenizer=eric_tt.tokenizer)

    args = TTTokArgs()
    t_tok_length(
        eric_model=eric_tt,
        input_data_list=[TT_DATA_JSONL],
        out_dir=TT_out_dir_DATA,
        args=args,
        target_length=target_length,
    )


def test_trunc():
    target_length = 4
    args = TTTokArgs(max_in_len=target_length, max_out_len=target_length)

    t_tok_length(
        eric_model=eric_tt,
        input_data_list=[TT_DATA_JSONL],
        out_dir=TT_out_dir_DATA,
        args=args,
        target_length=target_length,
    )
