from logging import Logger
from typing import List, Union, Optional

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from erictransformer.validator.tasks.task_validator import TaskValidator


class TCValidator(TaskValidator):
    def __init__(
        self,
        model_name: Union[str, PreTrainedModel, None],
        trust_remote_code: bool,
        tokenizer: Union[str, PreTrainedTokenizerBase],
        logger: Logger,
        labels: Optional[List[str]] = None
    ):
        self.labels = labels
        super().__init__(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            tokenizer=tokenizer,
            logger=logger,

        )

    def validate_init(self):

        if self.labels is not None:
            if type(self.labels) is not list:
                raise ValueError(
                    "self.labels is not a list of strings"
                )

            for label in self.labels:
                if type(label) is not str:
                    raise ValueError(
                        "self.labels is not a list of strings."
                    )

        super().validate_init()

    def validate_call(self, texts: Union[List[str], str], args=None):
        super().validate_call(texts, args)

