from typing import Union

from transformers import PreTrainedTokenizer

from erictransformer.eric_tasks.chat_stream_handlers.args import CHATStreamResult
from erictransformer.eric_tasks.chat_stream_handlers.stream_handler import (
    StreamHandler,
)


class DefaultStreamHandler(StreamHandler):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer=tokenizer)

    def step(self, token_str: str) -> Union[None, CHATStreamResult]:
        if not token_str:
            return None

        return CHATStreamResult(text=token_str, marker="text", payload={})
