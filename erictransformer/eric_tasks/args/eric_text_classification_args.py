from dataclasses import dataclass

from erictransformer.args import CallArgs, TokArgs


@dataclass(kw_only=True)
class TCTokArgs(TokArgs):
    max_len: int = -1


@dataclass(kw_only=True)
class TCCallArgs(CallArgs):
    pass
