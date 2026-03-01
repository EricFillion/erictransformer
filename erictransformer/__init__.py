from erictransformer.eric_tasks import (
    CHATCallArgs,
    CHATResult,
    CHATStreamResult,
    CHATTokArgs,
    GENCallArgs,
    GENResult,
    GENTokArgs,
    EricChat,
    EricGeneration,
    EricTextClassification,
    EricTextToText,
    TCCallArgs,
    TCResult,
    TCTokArgs,
    TTCallArgs,
    TTResult,
    TTTokArgs
)

from erictransformer.args import EricTrainArgs, EricEvalArgs
from erictransformer.eric_transformer import EricTransformer
from erictransformer.loops import EvalResult, TrainResult

__all__ = []

try:
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import save_model
    mlx_enabled = True
except ImportError as err:
    mlx_enabled = False

if mlx_enabled:
    try:
        from .eric_tasks.eric_chat_mlx import EricChatMLX
    except Exception:
        # Optional: log or ignore
        pass
    else:
        __all__.append("EricChatMLX")

name = "erictransformer"
