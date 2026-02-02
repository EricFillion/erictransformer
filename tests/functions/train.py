import copy
from typing import List, Union
import pathlib
import sys
import pytest

from erictransformer import EricTransformer
from erictransformer.args import EricEvalArgs, EricTrainArgs, CallArgs

train_loop_module = sys.modules["erictransformer.loops.train_loop"]

def t_train_basic(
    eric_model: EricTransformer, train_args: EricTrainArgs, train_eval_data: str
):
    # Smoke test.
    eric_model.train(train_eval_data, eval_path=train_eval_data, args=train_args)


def t_train_eff(
    eric_model: EricTransformer,
    train_args: EricTrainArgs,
    eval_args: EricEvalArgs,
    train_eval_data: str,
    call_text: str,
    call_args: Union[CallArgs, None] = None
):
    # Assert train decreases loss.

    before_result = eric_model.eval(train_eval_data, args=eval_args).loss
    if call_args is not None:
        output_before = eric_model(call_text, args=call_args)
    else:
        output_before = eric_model(call_text)

    train_args_copy = copy.deepcopy(train_args)
    train_args_copy.epochs = 10
    train_args_copy.lr = 1e-4
    eric_model.train(train_eval_data, args=train_args_copy)

    # we run __call__  before and after just to make sure nothing breaks from training
    if call_args is not None:
        output_after = eric_model(call_text, args=call_args)

    else:
        output_after = eric_model(call_text)

    after_result = eric_model.eval(train_eval_data, args=eval_args).loss

    print("after_result", after_result)
    print("before_result", before_result)

    assert after_result < before_result


def t_train_long(
    eric_model: EricTransformer,
    train_args: EricTrainArgs,
    eval_args: EricEvalArgs,
    train_eval_data: str,
):
    # Smoke test for longer train run.

    train_args_copy = copy.deepcopy(train_args)
    train_args_copy.lr = 0.01
    train_args_copy.eval_steps = 2
    train_args_copy.log_steps = 2
    train_args_copy.checkpoint_steps = 2
    train_args_copy.epochs = 5
    eric_model.train(
        train_eval_data, eval_path=train_eval_data, args=train_args_copy
    )


def t_train_formats(
    eric_model: EricTransformer, train_args: EricTrainArgs, train_data: List[str]
):
    # Smoke test for multiple import formats.

    for data in train_data:
        eric_model.train(data, eval_path=data, args=train_args)


def t_train_bs(
    eric_model: EricTransformer, train_args: EricTrainArgs, train_eval_data: str
):
    # Smoke test for non-default batch size.

    new_args = copy.deepcopy(train_args)
    new_args.bs = 2
    eric_model.train(train_eval_data, args=new_args)


def t_train_sgd(
    eric_model: EricTransformer, train_args: EricTrainArgs, train_eval_data: str
):
    # Smoke test for SGD optim.

    train_args_copy = copy.deepcopy(train_args)
    train_args_copy.optim = "sgd"
    eric_model.train(
        train_eval_data, eval_path=train_eval_data, args=train_args_copy
    )


def raise_error():
    raise Exception('fake exception')


def t_train_checkpoint(
    eric_model: EricTransformer, train_args: EricTrainArgs, train_eval_data: str
):

    train_args = copy.deepcopy(train_args)

    default_checkpoint_steps = train_args.checkpoint_steps
    train_args.checkpoint_steps = 1
    train_args.out_dir = str(pathlib.Path(train_args.out_dir) / 't_train_checkpoint')

    # Setup hook to track steps.
    steps = None
    def hook_steps(n):
        nonlocal steps
        steps = n

    # On first checkpoint, error after to interrupt.
    with train_loop_module.debug_hook_post_checkpoint.set(raise_error):
        train_args.run_name='run'
        # Run loop, expecting it to save a checkpoint and then get interrupted.
        with pytest.raises(Exception):
            eric_model.train(
                train_eval_data,
                eval_path=train_eval_data,
                args=train_args,
            )

    train_args.checkpoint_steps = default_checkpoint_steps

    # On resumed run, check steps at end to see what training actually did.
    with train_loop_module.debug_hook_steps.set(hook_steps):
        eric_model.train(
            train_eval_data,
            eval_path=train_eval_data,
            args=train_args,
            resume_path=str(
                pathlib.Path(train_args.out_dir)
                / 'run' / 'checkpoint'
            )
        )

    assert steps is not None
    assert 0 < steps["start_step"] < steps["total_steps"]

