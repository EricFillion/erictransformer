import json
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PretrainedConfig, PreTrainedTokenizerBase

from erictransformer.args.eric_args import EricTrainArgs
from erictransformer.eval_models import EvalModel
from erictransformer.exceptions import EricTrainError
from erictransformer.eric_tracker.eric_tracker import EricTracker
from erictransformer.loops.eval_loop import eval_loop
from erictransformer.utils import EricTimer
from erictransformer.utils.test import DebugHook


@dataclass
class TrainResult:
    final_train_loss: float
    final_eval_loss: Optional[float] = None
    best_eval_loss: Optional[float] = None


def train_loop(
    args: EricTrainArgs,
    train_dataloader: DataLoader,
    eric_tracker: EricTracker,
    train_steps: int,
    model: Module,
    optim: Optimizer,
    lr_sched: LRScheduler,
    eval_cases: int,
    eval_dataloader: DataLoader,
    tokenizer: PreTrainedTokenizerBase,
    config: PretrainedConfig,
    skip_cases: int,
    starting_epoch: int,
    eval_models: List[EvalModel],
    eric_timer: EricTimer,
    precision_type: str,
) -> TrainResult:
    best_tokenizer_config_saved = False
    checkpoint_tokenizer_config_saved = False
    first_epoch_train_iter = iter(train_dataloader)

    if skip_cases:
        try:
            skip_batches = max(1, int(skip_cases / args.bs))
            for _ in tqdm(range(skip_batches)):
                next(first_epoch_train_iter, None)
        except Exception as e:
            raise EricTrainError(f"Failed to skip initial training cases: {e}")

    device = next(model.parameters()).device

    gas = max(1, args.gas)

    start_step = eric_tracker.state.current_step

    for epoch in range(starting_epoch - 1, int(args.epochs)):
        epoch_iter = (
            first_epoch_train_iter
            if epoch == starting_epoch - 1
            else iter(train_dataloader)
        )

        batch_idx = 0
        while True:
            if eric_tracker.state.current_step > train_steps:
                break

            with eric_timer.section("training_core", "iter"):
                try:
                    batch = next(epoch_iter)
                except StopIteration:
                    break

            try:
                with eric_timer.section("training_core", "to_device"):
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(
                        device, non_blocking=True
                    )
                    labels = batch["labels"].to(device, non_blocking=True)
            except Exception as e:
                raise EricTrainError(f"Failed to extract inputs from batch: {e}")

            try:
                with eric_timer.section("training_core", "forward"):
                    if device.type == "cuda":
                        with torch.autocast(
                            device_type="cuda",
                            enabled=precision_type != "fp32",
                            dtype=(
                                torch.float16
                                if precision_type == "fp16"
                                else torch.bfloat16
                            ),
                        ):
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                            )
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )

                with eric_timer.section("training_core", "get loss"):
                    raw_loss = outputs.loss.detach()
                    loss = outputs.loss / gas

            except Exception as e:
                raise EricTrainError(f"Model forward pass failed: {e}")

            try:
                with eric_timer.section("training_core", "backwards"):
                    loss.backward()
            except Exception as e:
                raise EricTrainError(f"Backward pass failed: {e}")

            try:
                do_step = ((batch_idx + 1) % gas == 0) or (
                    (batch_idx + 1) == len(train_dataloader)
                )

                if do_step:
                    if precision_type == "fp16":
                        # for now we don't clip gradients when using bf16 or fp32 since this
                        # operation takes a noticeable amount of time and they're stable enough.
                        with eric_timer.section("training_core", "clip gradients"):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    with eric_timer.section("training_core", "optim.step()"):
                        optim.step()
                    with eric_timer.section("training_core", "lr_sched.step()"):
                        lr_sched.step()
                    with eric_timer.section("training_core", "optim.zero_grad()"):
                        optim.zero_grad()
                    eric_tracker.optim_step()

            except Exception as e:
                raise EricTrainError(f"optim or LR scheduler step failed: {e}")

            try:
                do_eval = bool(eric_tracker.time_to_eval() and eval_cases)

                if do_eval:
                    with eric_timer.section("training_extra", "eval loop"):
                        model.eval()
                        eval_result = eval_loop(
                            model,
                            eval_dataloader,
                            eval_cases,
                            args.eval_bs
                            if args.eval_bs
                            else args.bs,
                            eval_models=eval_models,
                        )
                        model.train()

                    global_eval_loss = float(eval_result.loss)

                    is_best_model = eric_tracker.set_eval_loss(global_eval_loss)
                    eric_tracker.set_metrics(eval_result.metrics)

                    if args.save_best and is_best_model:
                        with eric_timer.section("training_extra", "save best model"):
                            try:
                                path = eric_tracker.tracker_paths.best_model_path
                                model.save_pretrained(path)
                            except Exception as e:
                                path = eric_tracker.tracker_paths.best_model_path
                                raise EricTrainError(
                                    f"Failed to save best model to: {path} |  {e}"
                                )

                            try:
                                state_path = (
                                    eric_tracker.tracker_paths.best_model_state_path
                                )
                                with open(state_path, "w") as f:
                                    json.dump(
                                        eric_tracker.state.to_dict(), f, indent=2
                                    )
                            except Exception as e:
                                state_path = (
                                    eric_tracker.tracker_paths.best_model_state_path
                                )
                                raise EricTrainError(
                                    f"Failed to save best model state to: {state_path} | {e}"
                                )

                            if not best_tokenizer_config_saved:
                                try:
                                    tokenizer.save_pretrained(
                                        eric_tracker.tracker_paths.best_model_path
                                    )
                                except Exception as e:
                                    path = eric_tracker.tracker_paths.best_model_path
                                    raise EricTrainError(
                                        f"Failed to save tokenizer to: {path} | {e}"
                                    )
                                try:
                                    config.save_pretrained(
                                        eric_tracker.tracker_paths.best_model_path
                                    )
                                except Exception as e:
                                    path = eric_tracker.tracker_paths.best_model_path
                                    raise EricTrainError(
                                        f"Failed to save config to: {path} | {e}"
                                    )
                                best_tokenizer_config_saved = True
            except Exception as e:
                raise EricTrainError(f"Evaluation loop failed: {e}")

            try:
                do_ckpt = eric_tracker.time_to_checkpoint()

                if do_ckpt:
                    with eric_timer.section("training_extra", "checkpoint"):
                        checkpoint_path = eric_tracker.tracker_paths.checkpoint_path
                        try:
                            model.save_pretrained(checkpoint_path)
                        except Exception as e:
                            raise EricTrainError(
                                f"Failed to save checkpoint model to: {checkpoint_path} | {e}"
                            )

                        if not checkpoint_tokenizer_config_saved:
                            try:
                                tokenizer.save_pretrained(checkpoint_path)
                            except Exception as e:
                                raise EricTrainError(
                                    f"Failed to save tokenizer to: {checkpoint_path} |{e}"
                                )
                            try:
                                config.save_pretrained(checkpoint_path)
                            except Exception as e:
                                raise EricTrainError(
                                    f"Failed to save config to: {checkpoint_path} | {e}"
                                )
                            checkpoint_tokenizer_config_saved = True

                        try:
                            # Reloading the optimizer was causing problems. So for now we restart it.
                            #torch.save(
                            #     optim.state_dict(),
                            #     eric_tracker.tracker_paths.optim_path,
                            #  )

                            torch.save(
                                lr_sched.state_dict(),
                                eric_tracker.tracker_paths.lr_sched_path,
                            )

                        except Exception as e:
                            raise EricTrainError(
                                f"Failed to save lr scheduler state to: "
                                f"{eric_tracker.tracker_paths.optim_path} |  {e}"
                            )
            except Exception as e:
                raise EricTrainError(str(e))

            try:
                with eric_timer.section("training_extra", "logging"):
                    if (
                        eric_tracker.time_to_log()
                        or eric_tracker.time_to_eval()
                        or eric_tracker.time_to_checkpoint()
                    ):
                        eric_timer.report()

                    num_tokens = int(attention_mask.sum().item())
                    eric_tracker.step(
                        raw_loss, lr_sched.get_last_lr()[0], num_tokens=num_tokens
                    )
            except Exception as e:
                raise EricTrainError(f"Tracker step failed: {e}")

            debug_hook_post_checkpoint()

            batch_idx += 1

        eric_tracker.mark_epoch()

        if eric_tracker.state.current_step >= train_steps:
            break

    debug_hook_steps({
        "start_step": start_step,
        "total_steps": eric_tracker.state.current_step
    })

    return TrainResult(
        final_train_loss=eric_tracker.state.train_loss,
        final_eval_loss=eric_tracker.state.eval_loss if eval_cases else None,
        best_eval_loss=eric_tracker.state.best_eval_loss if eval_cases else None,
    )

debug_hook_post_checkpoint = DebugHook()
debug_hook_steps = DebugHook()
