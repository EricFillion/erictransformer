from dataclasses import dataclass, field
from ericsearch import SearchCallArgs

from erictransformer.args import CallArgs, TokArgs


@dataclass(kw_only=True)
class CHATCallArgs(CallArgs):
    min_len: int = 1
    max_len: int = 4096
    temp: float = 0.8
    top_k: int = 32
    top_p: float = 0.6

    # dataset args
    search_args: SearchCallArgs = field(default_factory=SearchCallArgs)


@dataclass(kw_only=True)
class CHATTokArgs(TokArgs):
    max_len: int = -1
