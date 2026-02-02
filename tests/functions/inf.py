import copy

from transformers import PreTrainedTokenizer

from erictransformer import EricTransformer
from erictransformer.args import CallArgs


def _check_length_and_type(
    text: str, tokenizer: PreTrainedTokenizer, token_length: int, is_text_to_text: bool
):
    assert type(text) == str
    tokens = tokenizer.encode(text, return_tensors="pt")
    print(text)
    length = tokens.size(1)
    print(length,  token_length)
    assert length == token_length


def t_inf_simple_chat_gen_wp_tt(
    eric_model: EricTransformer,
    call_args: CallArgs,
    call_text: str,
    output_length: int,
    is_text_to_text: bool = False,
):
    # str
    call_output = eric_model(call_text, args=call_args)
    call_str = call_output.text

    _check_length_and_type(
        call_str, eric_model.tokenizer, output_length, is_text_to_text
    )


def t_inf_algos(eric_model: EricTransformer, call_args: CallArgs, call_text: str):
    greedy_settings = copy.deepcopy(call_args)
    output_greedy = eric_model(call_text, args=greedy_settings)

    generic_sampling_settings = copy.deepcopy(call_args)
    generic_sampling_settings.top_k = 0
    generic_sampling_settings.temp = 0.7

    output_generic_sampling = eric_model(call_text, args=generic_sampling_settings)

    top_k_sampling_settings = copy.deepcopy(call_args)

    top_k_sampling_settings.top_k = 50
    top_k_sampling_settings.temp = 0.7

    output_top_k_sampling = eric_model(call_text, args=top_k_sampling_settings)

    top_p_sampling_settings = copy.deepcopy(call_args)
    top_p_sampling_settings.top_k = 0
    top_p_sampling_settings.top_p = 0.8
    top_p_sampling_settings.temp = 0.7

    output_top_p_sampling = eric_model(call_text, args=top_p_sampling_settings)

    assert type(output_greedy.text) == str
    assert type(output_generic_sampling.text) == str
    assert type(output_top_k_sampling.text) == str
    assert type(output_top_p_sampling.text) == str
