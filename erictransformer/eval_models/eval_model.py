from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
from transformers.modeling_outputs import ModelOutput

from erictransformer.exceptions import EricEvalModelError


class EvalModel(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def __call__(self, batch: Dict[str, torch.Tensor], outputs: ModelOutput) -> None:
        pass

    @abstractmethod
    def result(self) -> Optional[float]:
        pass

    @abstractmethod
    def _confirm_batch_params(
        self, batch: Dict[str, torch.Tensor], outputs: ModelOutput
    ) -> None:
        pass


class TCAccuracyEvalModel(EvalModel):
    def __init__(self):
        self.correct_preds = 0
        self.total_preds = 0
        self.name = "accuracy"
        super().__init__(self.name)

    def reset(self) -> None:
        self.correct_preds = 0
        self.total_preds = 0

    def __call__(self, batch: Dict[str, torch.Tensor], outputs: ModelOutput) -> None:
        try:
            self._confirm_batch_params(batch, outputs)
            logits = outputs.logits
            labels = batch["labels"]
            preds = torch.argmax(logits, dim=-1)
            self.correct_preds += (preds == labels).sum().item()
            self.total_preds += labels.size(0)
        except Exception as e:
            raise EricEvalModelError(f"error calling {self.__class__.__name__}: {e}")

    def result(self) -> Optional[float]:
        return self.correct_preds / self.total_preds if self.total_preds > 0 else None

    def _confirm_batch_params(
        self, batch: Dict[str, torch.Tensor], outputs: ModelOutput
    ) -> None:
        if "labels" not in batch:
            raise EricEvalModelError("Batch must contain a 'labels' key.")
        if not isinstance(batch["labels"], torch.Tensor):
            raise EricEvalModelError("batch['labels'] must be a torch.Tensor.")
        if not hasattr(outputs, "logits"):
            raise EricEvalModelError("ModelOutput must have a 'logits' attribute.")
        if not isinstance(outputs.logits, torch.Tensor):
            raise EricEvalModelError("outputs.logits must be a torch.Tensor.")

        bs_logits = outputs.logits.shape[0]
        bs_labels = batch["labels"].shape[0]
        if bs_logits != bs_labels:
            raise EricEvalModelError(
                f"Mismatch in batch size: logits has {bs_logits}, labels has {bs_labels}"
            )
