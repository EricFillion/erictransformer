import sys
import copy

from tests.chat import CHAT_CALL_ARGS, CHAT_CALL_TEXT
from tests.functions.inf import t_inf_algos, t_inf_simple_chat_gen_wp_tt
if sys.platform in ("darwin"):  # only for mac
    from tests.chat_mlx import eric_chat_mlx

def test_special():
    if sys.platform in ("darwin"):  # only for mac

        # list of dictionaries
        input_list_of_dict = [
            {"role": "system", "content": "You are a an AI assistant called Eric Transformer."},
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "5"},
        ]

        output_dict = eric_chat_mlx(input_list_of_dict, args=CHAT_CALL_ARGS)
        assert type(output_dict.text) == str

        # single string
        input_string = "Hi!"
        output_string = eric_chat_mlx(input_string, args=CHAT_CALL_ARGS)
        assert type(output_string.text) == str


def test_inf_basic():
    if sys.platform in ("darwin"):  # only for mac
        args = copy.deepcopy(CHAT_CALL_ARGS)
        args.max_len = 256 # to account for thinking tokens
        output = eric_chat_mlx(CHAT_CALL_TEXT, args=args)

        assert type(output.text) == str

def test_inf_algos():
    if sys.platform in ("darwin"):  # only for mac

        t_inf_algos(
            eric_model=eric_chat_mlx, call_args=CHAT_CALL_ARGS, call_text=CHAT_CALL_TEXT
        )
