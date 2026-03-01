import copy
from typing import List


from erictransformer import EvalResult, EricTransformer
from erictransformer.args import EricEvalArgs


def t_eval_formats(
    eric_model: EricTransformer, eval_args: EricEvalArgs, eval_data: List[str]
):
    for data in eval_data:
        result = eric_model.eval(data, args=eval_args)
        assert type(result) == EvalResult
        assert type(result.loss) == float


def t_eval_bs(eric_model: EricTransformer, eval_args: EricEvalArgs, eval_data: str):
    eval_args_copy = copy.deepcopy(eval_args)
    eval_args_copy.bs = 2
    result = eric_model.eval(eval_data, args=eval_args_copy)
    assert type(result) == EvalResult
    assert type(result.loss) == float
