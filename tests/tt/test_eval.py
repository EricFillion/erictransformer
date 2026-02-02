from tests.functions.eval import t_eval_bs, t_eval_formats
from tests.tt import TT_DATA_JSONL, TT_EVAL_ARGS, eric_tt


def test_eval_formats():
    data = [TT_DATA_JSONL]
    t_eval_formats(eric_tt, eval_args=TT_EVAL_ARGS, eval_data=data)


def test_eval_bs():
    t_eval_bs(eric_tt, eval_args=TT_EVAL_ARGS, eval_data=TT_DATA_JSONL)
