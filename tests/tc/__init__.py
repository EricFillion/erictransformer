from erictransformer import EricTextClassification, EricTrainArgs, EricEvalArgs
from tests.get_test_models import (
    PROVIDED_LABELS,
    TC_MODEL_PATH_2L,
    TC_MODEL_PATH_3L,
    TC_MODEL_PATH_LABELS,
    TC_MODEL_PATH_PROVIDE_LABELS,
)

eric_tc = EricTextClassification(model_name=TC_MODEL_PATH_2L)
eric_tc_train = EricTextClassification(model_name=TC_MODEL_PATH_2L)
eric_tc_3 = EricTextClassification(model_name=TC_MODEL_PATH_3L, labels=["one", "two", "three"])

eric_tc_labels = EricTextClassification(
    model_name=TC_MODEL_PATH_2L, labels=PROVIDED_LABELS
)

eric_tc_default_labels = EricTextClassification(model_name=TC_MODEL_PATH_LABELS)

eric_tc_provided_labels = EricTextClassification(
    model_name=TC_MODEL_PATH_PROVIDE_LABELS
)
PROVIDED_LABELS = PROVIDED_LABELS

TC_DATA_JSONL = "tc/data/input/train-eval.jsonl"

TC_DATA_MULTI_JSONL = "tc/data/input/train-eval-labels.jsonl"

TC_out_dir_DATA = "outputs/tc/data"
TC_out_dir_RUN = "outputs/tc/run"
# we save all models to the same dir to overwrite each other to minimize memory usage
TC_out_dir_MODEL = "outputs/tc/model"


TC_TRAIN_ARGS = EricTrainArgs(
    save_best=False, checkpoint_steps=-1, out_dir=TC_out_dir_RUN
)

TC_EVAL_ARGS = EricEvalArgs(out_dir=TC_out_dir_RUN)

TC_CALL_TEXT = "Wow what a great place to eat"  # LABEL_1
