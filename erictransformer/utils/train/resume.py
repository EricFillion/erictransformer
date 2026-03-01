import json
import os
from pathlib import Path

from erictransformer.exceptions import EricResumeError


def resume_training(resume_path: str):
    tracker_state = _load_tracker_state(resume_path)
    args_dict = _load_train_args(resume_path)
    model_tokenizer_path, lr_sched_path = _resolve_training_paths(
        resume_path
    )

    return (
        tracker_state,
        args_dict,
        model_tokenizer_path,
        lr_sched_path,
    )


def _load_tracker_state(resume_path: str):
    logs_path = Path(os.path.join(resume_path, "eric_tracker_state.json"))
    try:
        if not logs_path.exists():
            raise EricResumeError(f"No log file found at {logs_path}.")

        with logs_path.open(encoding="utf-8") as f:
            tracker_state = json.load(f)

        if tracker_state is None:
            raise EricResumeError("Loaded tracker state is None.")

        return tracker_state
    except Exception as e:
        raise EricResumeError(f"Failed to load tracker state: {e}")


def _load_train_args(resume_path: str):
    train_args_path = Path(os.path.join(resume_path, "train_args.json"))
    try:
        if not train_args_path.exists():
            raise EricResumeError(f"No train_args.json found at {train_args_path}.")

        args_dict = json.loads(train_args_path.read_text(encoding="utf-8"))
        return args_dict
    except Exception as e:
        raise EricResumeError(f"Failed to load training arguments: {e}")


def _resolve_training_paths(resume_path: str):
    try:
        lr_sched_path = os.path.join(resume_path, "lr_sched.pt")

        if not Path(lr_sched_path).exists():
            raise EricResumeError(f"lr_sched.pt not found at {lr_sched_path}")


        return resume_path, lr_sched_path
    except Exception as e:
        raise EricResumeError(f"Failed to resolve training paths: {e}")
