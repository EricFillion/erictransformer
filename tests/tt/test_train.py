from tests.functions.train import (
    t_train_basic,
    t_train_bs,
    t_train_eff,
    t_train_formats,
    t_train_long,
    t_train_checkpoint
)
from tests.tt import (
    TT_CALL_TEXT,
    TT_DATA_JSONL,
    TT_EVAL_ARGS,
    TT_TRAIN_ARGS,
    eric_tt,
    eric_tt_train,
)


def test_train_basic():
    # just runs train
    t_train_basic(
        eric_model=eric_tt, train_args=TT_TRAIN_ARGS, train_eval_data=TT_DATA_JSONL
    )



def test_train_eff():
    t_train_eff(
        eric_model=eric_tt_train,
        train_args=TT_TRAIN_ARGS,
        eval_args=TT_EVAL_ARGS,
        train_eval_data=TT_DATA_JSONL,
        call_text=TT_CALL_TEXT,
    )


def test_train_formats():
    train_data_list = [TT_DATA_JSONL]
    t_train_formats(
        eric_model=eric_tt, train_args=TT_TRAIN_ARGS, train_data=train_data_list
    )


def test_train_bs():
    t_train_bs(
        eric_model=eric_tt, train_args=TT_TRAIN_ARGS, train_eval_data=TT_DATA_JSONL
    )


def test_train_long():
    t_train_long(
        eric_model=eric_tt,
        train_args=TT_TRAIN_ARGS,
        eval_args=TT_EVAL_ARGS,
        train_eval_data=TT_DATA_JSONL,
    )


def test_train_checkpoint():
    # just runs train
    t_train_checkpoint(
        eric_model=eric_tt, train_args=TT_TRAIN_ARGS, train_eval_data=TT_DATA_JSONL
    )
