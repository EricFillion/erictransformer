import torch
from torch import device

from erictransformer.exceptions import EricDeviceError


def et_get_device() -> device:
    try:
        d = None
        if torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                d = torch.device("mps")

        if torch.cuda.is_available():
            d = torch.device("cuda:0")

        if not d:
            d = torch.device("cpu")

        return d
    except Exception as e:
        raise EricDeviceError(f"Device selection failed: {e}")
