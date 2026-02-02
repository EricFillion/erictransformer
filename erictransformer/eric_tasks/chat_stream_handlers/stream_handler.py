from abc import ABC, abstractmethod
from typing import Union

from transformers import PreTrainedTokenizer

from erictransformer.eric_tasks.chat_stream_handlers.args import CHATStreamResult


class StreamHandler(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.special_tokens = tokenizer.all_special_tokens
        self.eos_token = self.tokenizer.eos_token

    @abstractmethod
    def step(self, token_str: str) -> Union[None, CHATStreamResult]:
        pass
