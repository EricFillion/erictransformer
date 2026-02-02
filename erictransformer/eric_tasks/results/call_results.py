from dataclasses import dataclass
from typing import List

from erictransformer.args import CallResult


@dataclass(kw_only=True)
class GENResult(CallResult):
    text: str


@dataclass(kw_only=True)
class CHATResult(CallResult):
    text: str


@dataclass(kw_only=True)
class StreamCHATResult(CallResult):
    text: str
    mode: str


@dataclass(kw_only=True)
class TCResult(CallResult):
    labels: List[str]
    scores: List[float]

@dataclass(kw_only=True)
class TTResult(CallResult):
    text: str
