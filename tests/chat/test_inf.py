from tests.chat import CHAT_CALL_ARGS, CHAT_CALL_TEXT, eric_chat
from tests.functions.inf import t_inf_algos, t_inf_simple_chat_gen_wp_tt


def test_special():
    # list of dictionaries
    input_list_of_dict = [
        {"role": "system", "content": "You are a an AI assistant called Eric Transformer."},
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What's 2+2?"},
        {"role": "assistant", "content": "5"},
    ]

    output_dict = eric_chat(input_list_of_dict, args=CHAT_CALL_ARGS)
    assert type(output_dict.text) == str

    # single string
    input_string = "Hi!"
    output_string = eric_chat(input_string, args=CHAT_CALL_ARGS)
    assert type(output_string.text) == str


def test_inf_basic():
    t_inf_simple_chat_gen_wp_tt(
        eric_model=eric_chat,
        call_args=CHAT_CALL_ARGS,
        call_text=CHAT_CALL_TEXT,
        output_length=CHAT_CALL_ARGS.max_len,
    )


def test_inf_algos():
    t_inf_algos(
        eric_model=eric_chat, call_args=CHAT_CALL_ARGS, call_text=CHAT_CALL_TEXT
    )
