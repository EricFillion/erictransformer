import os
from typing import List

from erictransformer import EricTransformer
from erictransformer.args import EricEvalArgs, EricTrainArgs


def t_tok(
    eric_model: EricTransformer,
    input_data_list: List[str],
    out_dir: str,
    train_args: EricTrainArgs,
    eval_args: EricEvalArgs,
):
    out_dir = os.path.join(out_dir, "csv")

    for data in input_data_list:
        eric_model.tok(path=data, out_dir=out_dir)
        train_result = eric_model.train(
            train_path=out_dir, eval_path=out_dir, args=train_args
        )
        eval_result = eric_model.eval(eval_path=out_dir, args=eval_args)
