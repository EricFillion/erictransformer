import copy
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Optional

from tqdm.auto import tqdm

from erictransformer.args.eric_args import EricTrainArgs
from erictransformer.exceptions import EricIOError
from erictransformer.eric_tracker.save_plot import (
    save_lr_plot,
    save_metric_plots,
    save_train_eval_loss_plot,
)
from erictransformer.utils import make_dir


@dataclass
class TrackerState:
    current_step: int = 0
    last_eval_step: int = 0
    original_eval_loss: float = 0.0
    eval_loss: float = 0.0
    best_eval_loss: Optional[float] = None
    eval_loss_improvement: float = 0.0
    train_loss: float = 0.0
    last_checkpoint_step: Optional[int] = None
    epoch: int = 1
    lr: float = 0
    metrics: Optional[dict] = None
    num_tokens: int = 0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class TrackerPaths:
    log_path: str
    best_model_path: str
    train_args_path: str
    checkpoint_path: str
    checkpoint_state_path: str
    best_model_state_path: str
    lr_sched_path: str
    optim_path: str


class EricTracker:
    def __init__(
        self, args: EricTrainArgs, train_steps: int, out_dir: str, tracker_state
    ):
        self.train_args = args
        self.train_steps = train_steps

        self.out_dir = out_dir

        self.tracker_paths = self.__init_dirs(out_dir)
        self.log_steps, self.eval_steps, self.checkpoint_steps = (
            self.__get_log_eval_checkpoint_steps(self.train_args)
        )
        self._write_train_args()

        self.tracker_state_history = []

        if tracker_state is not None:
            self.state = TrackerState(**tracker_state)
            self.state.current_step += 1

        else:
            self.state = TrackerState()

        self.epoch_steps = []
        self.eval_steps_history = []
        self.pbar = tqdm(
            total=train_steps, initial=self.state.current_step, desc="Model Training"
        )

    def __init_dirs(self, out_dir):
        try:
            best_model_path = os.path.join(out_dir, "best_model")
            make_dir(best_model_path)

            best_model_state_path = os.path.join(
                best_model_path, "eric_tracker_state.json"
            )
            open(best_model_state_path, "a").close()

            log_path = os.path.join(out_dir, "logs.jsonl")
            open(log_path, "a").close()

            checkpoint_path = os.path.join(out_dir, "checkpoint")
            make_dir(checkpoint_path)

            train_args_path = os.path.join(out_dir, "checkpoint/train_args.json")
            open(train_args_path, "a").close()

            checkpoint_state_path = os.path.join(
                checkpoint_path, "eric_tracker_state.json"
            )
            open(checkpoint_state_path, "a").close()

            lr_sched_path = os.path.join(checkpoint_path, "lr_sched.pt")
            open(lr_sched_path, "a").close()

            optim_path = os.path.join(checkpoint_path, "optim.pt")
            open(optim_path, "a").close()
        except Exception as e:
            raise EricIOError(
                f"Error initializing tracker directories or files: {str(e)}"
            )

        return TrackerPaths(
            best_model_state_path=best_model_state_path,
            best_model_path=best_model_path,
            log_path=log_path,
            train_args_path=train_args_path,
            checkpoint_path=checkpoint_path,
            checkpoint_state_path=checkpoint_state_path,
            lr_sched_path=lr_sched_path,
            optim_path=optim_path,
        )

    def __get_log_eval_checkpoint_steps(self, args: EricTrainArgs):
        def resolve_steps(value):
            if value >= 1:
                return math.ceil(value)
            return 0

        return (
            resolve_steps(args.log_steps) or 1,
            resolve_steps(args.eval_steps),
            resolve_steps(args.checkpoint_steps),
        )

    def _write_train_args(self):
        try:
            with open(self.tracker_paths.train_args_path, "w") as f:
                json.dump(asdict(self.train_args), f, indent=2)
        except Exception as e:
            raise EricIOError(
                f"Error writing train args to {self.tracker_paths.train_args_path}: {e}"
            )

    def time_to_eval(self):
        return bool(
            self.eval_steps
            and (
                self.state.current_step % self.eval_steps == 0
                or self.state.current_step == 0
                or self.state.current_step + 1 == self.train_steps
            )
        )

    def time_to_checkpoint(self):
        if not self.checkpoint_steps:
            return False

        return (
            self.state.current_step % self.checkpoint_steps == 0
            or self.state.current_step + 1 == self.train_steps
        )

    def time_to_log(self):
        return (
            self.state.current_step % self.log_steps == 0
            or self.state.current_step == 0
            or self.state.current_step + 1 == self.train_steps
        )

    def set_train_loss(self, loss: float):
        self.state.train_loss = loss

    def set_eval_loss(self, eval_loss):
        is_best_model = False

        if self.state.current_step == 0:
            self.state.original_eval_loss = eval_loss

        self.state.eval_loss = eval_loss

        if self.state.best_eval_loss is None or eval_loss < self.state.best_eval_loss:
            self.state.best_eval_loss = eval_loss
            is_best_model = True
        self.state.eval_loss_improvement = self.state.original_eval_loss - eval_loss
        self.state.last_eval_step = self.state.current_step

        self.eval_steps_history.append(self.state.current_step)

        return is_best_model

    def set_metrics(self, metrics: dict):
        self.state.metrics = metrics

    def mark_epoch(self):
        self.epoch_steps.append(self.state.current_step)
        self.state.epoch += 1

    def step(self, loss: float, lr: float, num_tokens: int):
        self.state.lr = lr
        self.state.num_tokens += num_tokens

        if self.time_to_checkpoint():
            try:
                with open(self.tracker_paths.checkpoint_state_path, "w") as f:
                    self.state.last_checkpoint_step = self.state.current_step
                    json.dump(self.state.to_dict(), f, indent=2)
            except Exception as e:
                raise EricIOError(
                    f"Error writing checkpoint to {self.tracker_paths.checkpoint_state_path}: {e}"
                )

        if self.time_to_log() or self.time_to_checkpoint():
            self.set_train_loss(loss.item())
            self.tracker_state_history.append(copy.deepcopy(self.state))
            self._save_state(self.tracker_paths.log_path)
            save_train_eval_loss_plot(
                self.tracker_state_history, self.eval_steps_history, self.out_dir
            )
            save_lr_plot(self.tracker_state_history, self.out_dir)
            if self.eval_steps_history:  # if eval has happened
                save_metric_plots(self.tracker_state_history, self.out_dir)
            self.pbar.set_postfix(
                last_eval_step=self.state.last_eval_step,
                eval_loss_improvement=f"{self.state.eval_loss_improvement:.4f}",
                eval_loss=self.state.eval_loss,
                loss=f"{loss:.4f}",
                num_tokens=self.state.num_tokens,
            )

        self.state.current_step += 1

    def optim_step(self):
        self.pbar.update(1)
        self.pbar.refresh()  # force update

    def close(self):
        self.pbar.close()

    def _save_state(self, save_path):
        try:
            with open(save_path, "a") as f:
                to_save = self.state.to_dict()
                del to_save[
                    "original_eval_loss"
                ]  # deleted this since it's static. Maybe we should include it anyways
                json.dump(to_save, f, separators=(",", ":"), allow_nan=True)
                f.write("\n")
        except Exception as e:
            raise EricIOError(f"Error writing state to {save_path} {e}")
