from tests.functions.train import (
    t_train_basic,
    t_train_bs,
    t_train_eff,
    t_train_formats,
    t_train_long,
    t_train_checkpoint
)
from tests.tc import (
    TC_CALL_TEXT,
    TC_DATA_JSONL,
    TC_DATA_MULTI_JSONL,
    TC_EVAL_ARGS,
    TC_TRAIN_ARGS,
    eric_tc,
    eric_tc_provided_labels,
    eric_tc_train,
)


## Standard
def test_train_basic():
    # just runs train
    t_train_basic(
        eric_model=eric_tc, train_args=TC_TRAIN_ARGS, train_eval_data=TC_DATA_JSONL
    )


def test_train_eff():
    t_train_eff(
        eric_model=eric_tc_train,
        train_args=TC_TRAIN_ARGS,
        eval_args=TC_EVAL_ARGS,
        train_eval_data=TC_DATA_JSONL,
        call_text=TC_CALL_TEXT,
    )


def test_train_formats():
    train_data_list = [TC_DATA_JSONL]
    t_train_formats(
        eric_model=eric_tc, train_args=TC_TRAIN_ARGS, train_data=train_data_list
    )


def test_train_bs():
    t_train_bs(
        eric_model=eric_tc, train_args=TC_TRAIN_ARGS, train_eval_data=TC_DATA_JSONL
    )


def test_train_long():
    t_train_long(
        eric_model=eric_tc,
        train_args=TC_TRAIN_ARGS,
        eval_args=TC_EVAL_ARGS,
        train_eval_data=TC_DATA_JSONL,
    )


def test_train_provided_labels():
    t_train_eff(
        eric_model=eric_tc_provided_labels,
        train_args=TC_TRAIN_ARGS,
        eval_args=TC_EVAL_ARGS,
        train_eval_data=TC_DATA_MULTI_JSONL,
        call_text=TC_CALL_TEXT,
    )



def test_train_checkpoint():
    t_train_checkpoint(
        eric_model=eric_tc, train_args=TC_TRAIN_ARGS, train_eval_data=TC_DATA_JSONL
    )
