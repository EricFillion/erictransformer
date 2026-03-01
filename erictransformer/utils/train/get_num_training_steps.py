import math


def get_num_training_steps(
    train_cases: int,
    num_devices: int,
    epochs: float,
    gas: int,
    bs: int,
) -> int:
    steps = (train_cases * epochs) // (
        num_devices * gas * bs
    )

    return math.ceil(steps)
