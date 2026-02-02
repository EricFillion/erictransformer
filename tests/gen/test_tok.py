from tests.functions.tok import t_tok
from tests.gen import (
    GEN_DATA_DIR,
    GEN_DATA_JSONL,
    GEN_EVAL_ARGS,
    GEN_out_dir_DATA,
    GEN_TRAIN_ARGS,
    eric_gen,
)


def test_tok():
    # we expect this to fail until we update def tok() to take in specific files, rather than a dir.
    t_tok(
        eric_model=eric_gen,
        input_data_list=[GEN_DATA_JSONL, GEN_DATA_DIR],
        out_dir=GEN_out_dir_DATA,
        train_args=GEN_TRAIN_ARGS,
        eval_args=GEN_EVAL_ARGS,
    )
