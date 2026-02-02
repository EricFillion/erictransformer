from dataclasses import dataclass

@dataclass(kw_only=True)
class EricTrainArgs:
    # Learning parameters
    lr: float = 2e-5
    bs: int = 1
    eval_bs: int = 0  # when 0 uses bs
    epochs: int = 1
    gas: int = 1
    optim: str = "adamw"  # options adamw and sgd
    lr_sched: str = "constant"  # other option: warmup_then_decay

    # Action steps
    eval_steps: int = 256  # if 0 no evaluating will be done
    log_steps: int = 256  # if 0 no logging will be done
    checkpoint_steps: int = -1  # if -1 no checkpointing will be done
    save_best: bool = False  # saves the model with the lowest eval loss

    # Misc
    out_dir: str = "eric_transformer/"
    run_name: str = ""
    seed: int = 42


@dataclass(kw_only=True)
class TokArgs:
    bs: int = 1024
    max_cases: int = -1
    shards: int = 1
    procs: int = -1



@dataclass(kw_only=True)
class EricEvalArgs:
    bs: int = 1
    out_dir: str = "eric_transformer/"
    run_name: str = ""
    seed: int = 42


@dataclass(kw_only=True)
class CallArgs:
    pass


@dataclass(kw_only=True)
class CallResult:
    pass
