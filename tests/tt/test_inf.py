from tests.functions.inf import t_inf_algos, t_inf_simple_chat_gen_wp_tt
from tests.tt import TT_CALL_ARGS, TT_CALL_TEXT, eric_tt


def test_inf_basic():
    t_inf_simple_chat_gen_wp_tt(
        eric_model=eric_tt,
        call_args=TT_CALL_ARGS,
        call_text=TT_CALL_TEXT,
        output_length=TT_CALL_ARGS.max_len,
        is_text_to_text=True,
    )

def test_inf_algos():
    t_inf_algos(eric_model=eric_tt, call_args=TT_CALL_ARGS, call_text=TT_CALL_TEXT)
