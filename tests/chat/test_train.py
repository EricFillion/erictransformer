import copy

from tests.chat import (
    CHAT_CALL_TEXT,
    CHAT_DATA_JSONL,
    CHAT_EVAL_ARGS,
    CHAT_TRAIN_ARGS,
    eric_chat,
    eric_chat_train,
)
from tests.functions.train import (
    t_train_basic,
    t_train_bs,
    t_train_eff,
    t_train_formats,
    t_train_long,
    t_train_sgd,
    t_train_checkpoint
)


def test_train_basic():
    t_train_basic(
        eric_model=eric_chat,
        train_args=CHAT_TRAIN_ARGS,
        train_eval_data=CHAT_DATA_JSONL,
    )


def test_train_eff():
    t_train_eff(
        eric_model=eric_chat_train,
        train_args=CHAT_TRAIN_ARGS,
        eval_args=CHAT_EVAL_ARGS,
        train_eval_data=CHAT_DATA_JSONL,
        call_text=CHAT_CALL_TEXT,
    )


def test_train_formats():
    # trains with all file formats
    train_data_list = [CHAT_DATA_JSONL]
    t_train_formats(
        eric_model=eric_chat, train_args=CHAT_TRAIN_ARGS, train_data=train_data_list
    )


def test_train_bs():
    # increase batch size
    t_train_bs(
        eric_model=eric_chat,
        train_args=CHAT_TRAIN_ARGS,
        train_eval_data=CHAT_DATA_JSONL,
    )


def test_train_long():
    # long training run with multiple save/eval/log steps
    t_train_long(
        eric_model=eric_chat_train,
        train_args=CHAT_TRAIN_ARGS,
        eval_args=CHAT_EVAL_ARGS,
        train_eval_data=CHAT_DATA_JSONL,
    )


def test_train_sgd():
    t_train_sgd(
        eric_model=eric_chat,
        train_args=CHAT_TRAIN_ARGS,
        train_eval_data=CHAT_DATA_JSONL,
    )


def test_train_constant_lr():
    args = copy.deepcopy(CHAT_TRAIN_ARGS)
    args.lr_sched = "constant"
    t_train_basic(
        eric_model=eric_chat, train_args=args, train_eval_data=CHAT_DATA_JSONL
    )



def test_train_checkpoint():
    # just runs train
    t_train_checkpoint(
        eric_model=eric_chat, train_args=CHAT_TRAIN_ARGS, train_eval_data=CHAT_DATA_JSONL
    )
