from erictransformer import CHATCallArgs, EricEvalArgs, EricTrainArgs, EricChat
from tests.get_test_models import CHAT_MODEL_PATH

eric_chat = EricChat(model_name=CHAT_MODEL_PATH)
eric_chat_train = EricChat(model_name=CHAT_MODEL_PATH)

CHAT_DATA_JSONL = "chat/data/data.jsonl"

CHAT_out_dir_DATA = "outputs/chat/data"
CHAT_out_dir_RUN = "outputs/chat/run"
# we save all models to the same dir to overwrite each other to minimize memory usage
CHAT_out_dir_MODEL = "outputs/chat/model"


CHAT_TRAIN_ARGS = EricTrainArgs(
    save_best=False, checkpoint_steps=-1, out_dir=CHAT_out_dir_MODEL
)

CHAT_EVAL_ARGS = EricEvalArgs(out_dir=CHAT_out_dir_RUN)

CHAT_CALL_TEXT = "Tell me a joke"

CHAT_CALL_ARGS = CHATCallArgs(min_len=5, max_len=5, top_k=0, top_p=0.0)
