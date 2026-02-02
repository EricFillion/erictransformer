from erictransformer import TCTokArgs
from erictransformer.eric_tasks.tok.tok_functions import get_max_in_len
from tests.functions.tok_length import t_tok_length
from tests.tc import  TC_DATA_JSONL, TC_out_dir_DATA, eric_tc


def test_pad():
    target_length = 256

    args = TCTokArgs(max_len=target_length)

    t_tok_length(
        eric_model=eric_tc,
        input_data_list=[TC_DATA_JSONL],
        out_dir=TC_out_dir_DATA,
        args=args,
        target_length=target_length,
    )


def test_pad_max():
    target_length = get_max_in_len(max_len=-1, tokenizer=eric_tc.tokenizer)
    args = TCTokArgs()
    t_tok_length(
        eric_model=eric_tc,
        input_data_list=[TC_DATA_JSONL],
        out_dir=TC_out_dir_DATA,
        args=args,
        target_length=target_length,
    )


def test_trunc():
    target_length = 4
    args = TCTokArgs(max_len=target_length)

    t_tok_length(
        eric_model=eric_tc,
        input_data_list=[TC_DATA_JSONL],
        out_dir=TC_out_dir_DATA,
        args=args,
        target_length=target_length,
    )
