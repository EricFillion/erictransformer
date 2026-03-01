from erictransformer.eric_tasks.args import (
    CHATCallArgs,
    CHATTokArgs,
    GENCallArgs,
    GENTokArgs,
    TCCallArgs,
    TCTokArgs,
    TTCallArgs,
    TTTokArgs,
)
from erictransformer.eric_tasks.chat_stream_handlers import CHATStreamResult
from erictransformer.eric_tasks.eric_chat import EricChat
from erictransformer.eric_tasks.eric_generation import EricGeneration
from erictransformer.eric_tasks.eric_text_classification import (
    EricTextClassification,
)
from erictransformer.eric_tasks.eric_text_to_text import (
    EricTextToText,
    TTStreamResult,
)
from erictransformer.eric_tasks.results import (
    CHATResult,
    GENResult,
    TCResult,
    TTResult
)

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
        from .eric_chat_mlx import EricChatMLX
    except Exception:
        # Optional: log or ignore
        pass
    else:
        __all__.append("EricChatMLX")
