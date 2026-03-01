from erictransformer import GENTokArgs
from erictransformer.eric_tasks.tok.tok_functions import get_max_in_len
from tests.functions.tok_length import t_tok_length
from tests.gen import (
    GEN_DATA_JSONL,
    GEN_out_dir_DATA,
    eric_gen,
)


def test_pad():
    target_length = 256

    args = GENTokArgs(max_len=target_length)

    t_tok_length(
        eric_model=eric_gen,
        input_data_list=[GEN_DATA_JSONL],
        out_dir=GEN_out_dir_DATA,
        args=args,
        target_length=target_length,
    )


def test_pad_max():
    target_length = get_max_in_len(max_len=-1, tokenizer=eric_gen.tokenizer)
    args = GENTokArgs(max_len=target_length)
    t_tok_length(
        eric_model=eric_gen,
        input_data_list=[GEN_DATA_JSONL],
        out_dir=GEN_out_dir_DATA,
        args=args,
        target_length=target_length,
    )


def test_trunc():
    target_length = 4
    args = GENTokArgs(max_len=target_length)

    t_tok_length(
        eric_model=eric_gen,
        input_data_list=[GEN_DATA_JSONL],
        out_dir=GEN_out_dir_DATA,
        args=args,
        target_length=target_length,
    )
