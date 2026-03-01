from dataclasses import dataclass
from typing import Optional

@dataclass(kw_only=True)
class CHATStreamResult:
    text: str  # what user sees
    marker: str  # the marker e.g text, special, tool
    payload: Optional[dict]


@dataclass(kw_only=True)
class MarkerStrings:
    pass
