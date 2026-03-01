from abc import abstractmethod
from dataclasses import is_dataclass
from logging import Logger
from typing import Union

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from erictransformer.validator import EricValidator


class TaskValidator(EricValidator):
    def __init__(
        self,
        model_name: Union[str, PreTrainedModel, None],
        trust_remote_code: bool,
        tokenizer: Union[str, PreTrainedTokenizerBase],
        logger: Logger,
    ):
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.tokenizer = tokenizer
        self.logger = logger

        super().__init__()

    @abstractmethod
    def validate_init(self):
        self._validate_model_name()
        self._validate_tokenizer()

    def _validate_model_name(self):
        if not (
            isinstance(self.model_name, (str, PreTrainedModel))
            or self.model_name is None
        ):
            raise ValueError(
                "model_name must be a string, PreTrainedModel instance, or None."
            )

    def _validate_tokenizer(self):
        if self.tokenizer is not None:
            if not isinstance(self.tokenizer, (str, PreTrainedTokenizerBase)):
                print(self.tokenizer)
                raise ValueError(
                    "tokenizer must be a string, PreTrainedTokenizer, or PreTrainedTokenizerFast."
                )

    @abstractmethod
    def validate_call(self, text: str, args=None):
        if not isinstance(text, str):
            raise ValueError('"text" must be a string')

        if not is_dataclass(args):
            raise ValueError('"args" must be a dataclass')
