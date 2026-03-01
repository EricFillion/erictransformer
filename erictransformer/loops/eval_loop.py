import math
import sys
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.amp import autocast
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from erictransformer.eval_models import EvalModel
from erictransformer.exceptions import EricEvalError
from erictransformer.utils import get_precision


@dataclass(kw_only=True)
class EvalResult:
    loss: float
    metrics: Optional[dict] = None


def eval_loop(
    model: Module,
    dataloader: DataLoader,
    eval_tok_data_cases: int,
    eval_bs: int,
    eval_models: List[EvalModel],
    precision: str = "auto",
) -> EvalResult:
    total_loss = 0.0
    total_examples = 0
    pbar = tqdm(
        total=math.ceil(eval_tok_data_cases / eval_bs),
        desc="Model Evaluating",
        position=1,
        leave=False,
        disable=not sys.stdout.isatty(),
    )

    for eval_model in eval_models:
        eval_model.reset()

    model.eval()

    device = next(model.parameters()).device
    precision_type = get_precision(device=device)

    with torch.no_grad():
        for batch in dataloader:
            try:
                try:
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(
                        device, non_blocking=True
                    )
                    labels = batch["labels"].to(device, non_blocking=True)
                except Exception as e:
                    raise EricEvalError(f"Failed to extract inputs from batch: {e}")

                if device.type == "cuda":
                    dtype = (
                        torch.bfloat16 if precision_type == "bf16" else
                        torch.float16 if precision_type == "fp16" else
                        torch.float32
                    )
                    with autocast(device_type="cuda", enabled=(dtype != torch.float32), dtype=dtype):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                bs = batch["input_ids"].size(0)
                result = outputs.loss.item() * bs
            except Exception as e:
                raise EricEvalError(f"Error processing batch: {e}")

            if not math.isnan(result):
                total_loss += result
                total_examples += bs
            if eval_models:
                batch["input_ids"] = batch["input_ids"].to(device)
                batch["attention_mask"] = batch["attention_mask"].to(device)
                batch["labels"] = batch["labels"].to(device)

                for eval_model in eval_models:
                    eval_model(batch, outputs)

            pbar.update(1)

    pbar.close()

    custom_metrics = {}
    for eval_model in eval_models:
        name = eval_model.name
        if name in custom_metrics:
            for i in range(0, 1000):
                new_name = f"{eval_model.name}-{i}"
                if new_name not in custom_metrics:
                    name = new_name
                    break
            else:
                raise EricEvalError(
                    f"Couldn't find unique name for eval model named {eval_model.name}"
                )

        custom_metrics[name] = eval_model.result()
        eval_model.reset()

    average_loss = total_loss / total_examples if total_examples else 0.0

    return EvalResult(loss=average_loss, metrics=custom_metrics)
