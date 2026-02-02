import math
import os
from typing import Optional


def get_procs(procs: int) -> Optional[int]:
    if procs == -1:
        return None
    elif procs == 0:
        cpu_count = os.cpu_count()
        if cpu_count is None or cpu_count <= 1:
            return 1
        return math.floor(cpu_count / 2)

    return procs
