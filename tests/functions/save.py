from dataclasses import fields
from typing import Type, TypeVar, Union

from erictransformer import EricTransformer
from erictransformer.args import CallArgs

T = TypeVar("T", bound=EricTransformer)


def has_field(cls_or_instance, name: str) -> bool:
    return name in {f.name for f in fields(cls_or_instance)}


def t_save(
    eric_model: EricTransformer, eric_type: Type[T], call_input: str, save_path: str, args: Union[None, CallArgs]
):
    eric_model.save(save_path)
    if args is not None:
        result_before = eric_model(call_input, args=args)
    else:
        result_before = eric_model(call_input)

    eric_model_loaded = eric_type(model_name=save_path)

    if args is not None:
        result_after = eric_model_loaded(call_input, args=args)
    else:
        result_after = eric_model(call_input)

    print("result_before", result_before)
    print("result_after", result_after)

    if has_field(result_before, "text") and has_field(result_after, "text"):
        # GEN, Chat, TT
        assert result_before.text == result_after.text
    elif has_field(result_before, "labels") and has_field(result_after, "labels"):
        assert result_before.labels[0] == result_after.labels[0]
    else:
        assert False
