from dataclasses import dataclass
from typing import Union
import logging

from transformers import PreTrainedTokenizer

from erictransformer.eric_tasks.chat_stream_handlers.args import (
    CHATStreamResult,
    MarkerStrings,
)
from erictransformer.eric_tasks.chat_stream_handlers.stream_handler import (
    StreamHandler,
)


@dataclass(kw_only=True)
class SMOLMarkerStrings(MarkerStrings):
    think_start: str
    think_end: str
    tool_start: str
    tool_end: str


class SmolStreamHandler(StreamHandler):
    def __init__(self, tokenizer: PreTrainedTokenizer,  logger: logging.Logger):
        self.markers: SMOLMarkerStrings = SMOLMarkerStrings(
            think_start="<think>",
            think_end="</think>",
            tool_start="<tool_call>", # not supported
            tool_end="</tool_call>", # not supported
        )

        self.marker_key_tuple = (
            self.markers.think_start,
            self.markers.think_end,
            self.markers.tool_start,
            self.markers.tool_end,
        )

        self.in_thinking = False

        self.logger = logger

        super().__init__(tokenizer=tokenizer)

    def step(self, token_str: str) -> Union[None, CHATStreamResult]:
        if not token_str:
            return None

        stripped_token_str = token_str.strip()

        if stripped_token_str not in self.marker_key_tuple:
            if self.in_thinking:
                return CHATStreamResult(text=token_str, marker="thinking", payload={})

            if stripped_token_str in self.special_tokens:
                return CHATStreamResult(
                    text=stripped_token_str, marker="special", payload={})

            else:
                if token_str.endswith(self.eos_token):
                    token_str = token_str[: -len(self.eos_token)]
                return CHATStreamResult(text=token_str, marker="text", payload={})

        elif stripped_token_str == self.markers.think_start:
            self.in_thinking = True
            return CHATStreamResult(
                text=stripped_token_str, marker="think_start", payload={}
            )

        elif stripped_token_str == self.markers.think_end:
            self.in_thinking = False
            return CHATStreamResult(
                text=stripped_token_str, marker="think_end", payload={}
            )

        elif stripped_token_str in (self.markers.tool_start, self.markers.tool_end):
            self.logger.warning("Tool calling is not supported but a tool token was generated")
            return None

        return None
