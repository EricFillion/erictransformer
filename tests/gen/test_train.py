from tests.functions.train import (
    t_train_basic,
    t_train_bs,
    t_train_eff,
    t_train_formats,
    t_train_long,
    t_train_checkpoint
)
from tests.gen import (
    GEN_CALL_TEXT,
    GEN_DATA_JSONL,
    GEN_EVAL_ARGS,
    GEN_TRAIN_ARGS,
    eric_gen,
    eric_gen_train,
    GEN_CALL_ARGS
)


def test_train_basic():
    t_train_basic(
        eric_model=eric_gen, train_args=GEN_TRAIN_ARGS, train_eval_data=GEN_DATA_JSONL
    )



def test_train_eff():
    t_train_eff(
        eric_model=eric_gen_train,
        train_args=GEN_TRAIN_ARGS,
        eval_args=GEN_EVAL_ARGS,
        train_eval_data=GEN_DATA_JSONL,
        call_text=GEN_CALL_TEXT,
        call_args= GEN_CALL_ARGS
    )


def test_train_formats():
    # trains with all file formats
    train_data_list = [GEN_DATA_JSONL]
    t_train_formats(
        eric_model=eric_gen, train_args=GEN_TRAIN_ARGS, train_data=train_data_list
    )


def test_train_bs():
    # increase batch size
    t_train_bs(
        eric_model=eric_gen, train_args=GEN_TRAIN_ARGS, train_eval_data=GEN_DATA_JSONL
    )


def test_train_long():
    # long training run with multiple save/eval/log steps
    t_train_long(
        eric_model=eric_gen_train,
        train_args=GEN_TRAIN_ARGS,
        eval_args=GEN_EVAL_ARGS,
        train_eval_data=GEN_DATA_JSONL,
    )


def test_train_checkpoint():
    t_train_checkpoint(
        eric_model=eric_gen_train, train_args=GEN_TRAIN_ARGS, train_eval_data=GEN_DATA_JSONL
    )


