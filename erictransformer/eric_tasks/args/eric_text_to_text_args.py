from dataclasses import dataclass

from erictransformer.args import CallArgs, TokArgs


@dataclass(kw_only=True)
class TTCallArgs(CallArgs):
    min_len: int = 1
    max_len: int = 1024
    temp: float = 0.8
    top_k: int = 32
    top_p: float = 0.6


@dataclass(kw_only=True)
class TTTokArgs(TokArgs):
    max_in_len: int = -1
    max_out_len: int = -1
