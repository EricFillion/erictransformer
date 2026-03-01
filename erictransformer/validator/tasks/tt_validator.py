from logging import Logger
from typing import Union

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from erictransformer.validator.tasks.task_validator import TaskValidator


class TTValidator(TaskValidator):
    def __init__(
        self,
        model_name: Union[str, PreTrainedModel, None],
        trust_remote_code: bool,
        tokenizer: Union[str, PreTrainedTokenizerBase],
        logger: Logger,
    ):
        super().__init__(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            tokenizer=tokenizer,
            logger=logger,
        )

    def validate_init(self):
        super().validate_init()

    def validate_call(self, text: str, args=None):
        super().validate_call(text, args)
