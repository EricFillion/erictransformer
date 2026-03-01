from tests.functions.eval import t_eval_bs, t_eval_formats
from tests.tc import  TC_DATA_JSONL, TC_EVAL_ARGS, eric_tc


def test_eval_formats():
    data = [TC_DATA_JSONL]
    t_eval_formats(eric_tc, eval_args=TC_EVAL_ARGS, eval_data=data)


def test_eval_bs():
    t_eval_bs(eric_tc, eval_args=TC_EVAL_ARGS, eval_data=TC_DATA_JSONL)
