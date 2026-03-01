from typing import Union

from transformers import TextIteratorStreamer

from erictransformer.exceptions import EricInferenceError
from erictransformer.eric_tasks.args import CHATCallArgs, GENCallArgs, TTCallArgs


def format_messages(text):
    if isinstance(text, str):
        messages = [{"role": "user", "content": text}]
    elif isinstance(text, list) and all(isinstance(el, dict) for el in text):
        messages = text
    else:
        raise EricInferenceError("Wrong input format")

    return messages


def generate_gen_kwargs(
    input_ids,
    attention_mask,
    streamer: TextIteratorStreamer,
    args: Union[CHATCallArgs, GENCallArgs],
    eos_token_id: int,
    pad_token_id: int,
) -> dict:
    max_len = args.max_len
    if args.min_len > args.max_len:
        max_len = args.min_len

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        max_new_tokens=max_len,
        temp=args.temp,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    return gen_kwargs


def generate_tt_kwargs(
    input_ids,
    attention_mask,
    streamer: TextIteratorStreamer,
    args: Union[CHATCallArgs, TTCallArgs],
    eos_token_id: int,
    pad_token_id: int
) -> dict:
    max_len = args.max_len
    if args.min_len > args.max_len:
        max_len = args.min_len

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        max_new_tokens=max_len,
        temp=args.temp,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id
    )

    return gen_kwargs
