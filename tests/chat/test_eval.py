from tests.chat import CHAT_DATA_JSONL, CHAT_EVAL_ARGS, eric_chat
from tests.functions.eval import t_eval_bs, t_eval_formats


def test_eval_formats():
    data = [CHAT_DATA_JSONL]
    t_eval_formats(eric_chat, eval_args=CHAT_EVAL_ARGS, eval_data=data)


def test_eval_bs():
    t_eval_bs(eric_chat, eval_args=CHAT_EVAL_ARGS, eval_data=CHAT_DATA_JSONL)
