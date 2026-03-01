from erictransformer import EricTextToText, TTCallArgs, EricTrainArgs, EricEvalArgs
from tests.get_test_models import TT_MODEL_PATH

eric_tt = EricTextToText(model_name=TT_MODEL_PATH)

eric_tt_train = EricTextToText(model_name=TT_MODEL_PATH)

TT_DATA_JSONL = "tt/data/data.jsonl"

TT_out_dir_DATA = "outputs/tt/data"
TT_out_dir_RUN = "outputs/tt/run"
# we save all models to the same dir to overwrite each other to minimize memory usage
TT_out_dir_MODEL = "outputs/tt/model"


TT_TRAIN_ARGS = EricTrainArgs(
    save_best=False, checkpoint_steps=-1, out_dir=TT_out_dir_RUN
)

TT_EVAL_ARGS = EricEvalArgs(out_dir=TT_out_dir_RUN)

TT_CALL_TEXT = "Translate English to French: Hello"


TT_CALL_ARGS = TTCallArgs(min_len=5, max_len=5, top_k=0, top_p=0.0)
