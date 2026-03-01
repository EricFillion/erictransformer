from tests.functions.inf import t_inf_algos, t_inf_simple_chat_gen_wp_tt
from tests.gen import GEN_CALL_ARGS, GEN_CALL_TEXT, eric_gen


def test_inf_basic():
    t_inf_simple_chat_gen_wp_tt(
        eric_model=eric_gen,
        call_args=GEN_CALL_ARGS,
        call_text=GEN_CALL_TEXT,
        output_length=GEN_CALL_ARGS.max_len,
    )


def test_inf_algos():
    t_inf_algos(eric_model=eric_gen, call_args=GEN_CALL_ARGS, call_text=GEN_CALL_TEXT)
