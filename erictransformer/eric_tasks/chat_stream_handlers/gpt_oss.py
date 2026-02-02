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
class GPTOSSMarkerStrings(MarkerStrings):
    start: str = "<|start|>"
    end: str = "<|end|>"
    message: str = "<|message|>"
    channel: str = "<|channel|>"
    constrain: str = "<|constrain|>"
    return_token: str = "<|return|>"
    call: str = "<|call|>"


class GPTOSSSMHandler(StreamHandler):
    def __init__(self, tokenizer: PreTrainedTokenizer, logger: logging.Logger):
        self.markers = GPTOSSMarkerStrings()
        self.current_channel = ""
        self.in_thinking = False
        self.in_tool = False
        self.change_channel = False
        self.just_received_message = False
        self.start_just_happened = False
        self.tool_strings = []
        self.logger = logger
        super().__init__(tokenizer=tokenizer)

    def _reset_state(self):
        self.current_channel = ""
        self.in_thinking = False
        self.in_tool = False
        self.change_channel = False
        self.just_received_message = False
        self.start_just_happened = False
        self.tool_strings = []

    def step(self, token_str: str) -> Union[None, CHATStreamResult]:
        if not token_str:
            return None

        stripped_token_str = token_str.strip()

        self.in_thinking = False
        #### SPECIAL TOKENS ####

        # Case: <|start|>
        if stripped_token_str == self.markers.start:
            self._reset_state()
            self.start_just_happened = True
            return CHATStreamResult(
                text=stripped_token_str, marker="special", payload={}
            )

        # Case: <|channel|>
        if stripped_token_str == self.markers.channel:
            self.change_channel = True
            # next round
            return CHATStreamResult(
                text=stripped_token_str, marker="special", payload={}
            )

        # Case: <|message|>
        if stripped_token_str == self.markers.message:

            return CHATStreamResult(
                text=stripped_token_str, marker="special", payload={}
            )

        # Case: <|return|>
        if stripped_token_str == self.markers.return_token:
            # do nothing. Streaming is over.
            self._reset_state()
            return CHATStreamResult(
                text=stripped_token_str, marker="special", payload={}
            )

        # Case: <|constrain|>
        if stripped_token_str == self.markers.constrain:
            # do nothing. we always assume json
            return CHATStreamResult(
                text=stripped_token_str, marker="special", payload={}
            )

        # Case: <|end|>
        if stripped_token_str == self.markers.end:
            temp_current_channel = self.current_channel
            self._reset_state()
            if temp_current_channel == "analysis":
                return CHATStreamResult(
                    text=stripped_token_str, marker="think_end", payload={}
                )
            return CHATStreamResult(
                text=stripped_token_str, marker="special", payload={}
            )

        #### NON SPECIAL TOKENS ####

        if self.start_just_happened:
            # we don't do anything with this header token
            self.start_just_happened = False

            if stripped_token_str in {"assistant", "user", "system"}:
                return CHATStreamResult(text=token_str, marker="special", payload={})

        if self.change_channel:
            self.change_channel = False

            if self.current_channel == "commentary":
                self.logger.warning("The 'commentary' channel is not supported. Falling back to 'analysis'. ")
                self.current_channel = "analysis"

            if stripped_token_str not in ("analysis", "final"):
                # fall back to final
                self.logger.warning(f"Unexpected channel '{stripped_token_str}'. Falling back to 'final'.")
                self.current_channel = "final"
            else:
                self.current_channel = stripped_token_str

            if stripped_token_str == "analysis":
                return CHATStreamResult(
                    text=stripped_token_str, marker="think_start", payload={}
                )

            return CHATStreamResult(
                text=stripped_token_str, marker="special", payload={}
            )
        if self.current_channel == "analysis":
            # just return the text
            return CHATStreamResult(text=token_str, marker="thinking", payload={})

        if self.current_channel == "final":
            # just return the text
            return CHATStreamResult(text=token_str, marker="text", payload={})

        self._reset_state()
