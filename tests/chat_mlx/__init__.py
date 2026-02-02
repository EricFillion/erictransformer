import sys
if sys.platform == "darwin":
    from erictransformer import EricChatMLX
    from tests.get_test_models import CHAT_MODEL_PATH_MLX

    CHAT_MLX_out_dir_MODEL = "outputs/chat_mlx/model"

    eric_chat_mlx = EricChatMLX(model_name=CHAT_MODEL_PATH_MLX)
