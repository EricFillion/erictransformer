from erictransformer import TCTokArgs
from tests.functions import t_detok_jsonl, t_detok_len
from tests.tc import TC_DATA_JSONL, TC_out_dir_RUN, eric_tc



def test_detok_jsonl():
    t_detok_jsonl(
        eric_model=eric_tc, input_data=TC_DATA_JSONL, out_dir=TC_out_dir_RUN
    )


def test_detok_length():
    args = TCTokArgs()
    t_detok_len(
        eric_model=eric_tc,
        input_data=TC_DATA_JSONL,
        out_dir=TC_out_dir_RUN,
        args=args,
    )
