from erictransformer import  EricTrainArgs, EricEvalArgs, EricGeneration, GENCallArgs
from tests.get_test_models import GEN_MODEL_PATH

eric_gen = EricGeneration(model_name=GEN_MODEL_PATH)
eric_gen_train = EricGeneration(model_name=GEN_MODEL_PATH)


GEN_DATA_JSONL = "gen/data/train-eval.jsonl"


GEN_DATA_DIR = "gen/data/"

GEN_out_dir_DATA = "outputs/gen/data"
GEN_out_dir_RUN = "outputs/gen/run"
# we save all models to the same dir to overwrite each other to minimize memory usage
GEN_out_dir_MODEL = "outputs/gen/model"

GEN_TRAIN_ARGS = EricTrainArgs(
    save_best=False, checkpoint_steps=-1, out_dir=GEN_out_dir_RUN
)

GEN_EVAL_ARGS = EricEvalArgs(out_dir=GEN_out_dir_RUN)

GEN_CALL_TEXT = "Natural language processing is"

GEN_CALL_ARGS = GENCallArgs(min_len=5, max_len=5, top_k=0, top_p=0.0)
