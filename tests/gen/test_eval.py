from tests.functions.eval import t_eval_bs, t_eval_formats
from tests.gen import (
    GEN_DATA_JSONL,
    GEN_EVAL_ARGS,
    eric_gen,
)


def test_eval_formats():
    data = [GEN_DATA_JSONL]
    t_eval_formats(eric_gen, eval_args=GEN_EVAL_ARGS, eval_data=data)


def test_eval_bs():
    t_eval_bs(eric_gen, eval_args=GEN_EVAL_ARGS, eval_data=GEN_DATA_JSONL)
