from dataclasses import dataclass

from erictransformer.args import CallArgs, TokArgs


@dataclass(kw_only=True)
class GENTokArgs(TokArgs):
    max_len: int = 1024


@dataclass(kw_only=True)
class GENCallArgs(CallArgs):  # ← new canonical name
    min_len: int = 1
    max_len: int = 1024
    temp: float = 0.8
    top_k: int = 32
    top_p: float = 0.6



