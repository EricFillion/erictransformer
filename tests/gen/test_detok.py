from erictransformer import GENTokArgs
from tests.functions import  t_detok_jsonl, t_detok_len
from tests.gen import (
    GEN_DATA_JSONL,
    GEN_out_dir_DATA,
    eric_gen,
)



def test_detok_jsonl():
    t_detok_jsonl(
        eric_model=eric_gen, input_data=GEN_DATA_JSONL, out_dir=GEN_out_dir_DATA
    )


def test_detok_length():
    args = GENTokArgs()
    t_detok_len(
        eric_model=eric_gen,
        input_data=GEN_DATA_JSONL,
        out_dir=GEN_out_dir_DATA,
        args=args,
    )
