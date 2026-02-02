import sys
from sentence_transformers import CrossEncoder, SentenceTransformer

from erictransformer import (
    EricChat,
    EricGeneration,
    EricTextClassification,
    EricTextToText,
)

if sys.platform in ("darwin"):
    from erictransformer import EricChatMLX

#### Chat ####
CHAT_MODEL = "microsoft/DialoGPT-small"
CHAT_MODEL_PATH = "models/chat"

#### Chat MLX ####
CHAT_MODEL_MLX = "mlx-community/SmolLM3-3B-4bit"
CHAT_MODEL_PATH_MLX = "models/chat_mlx"

#### Text Generation ####
GEN_MODEL = "openai-community/gpt2"
GEN_MODEL_PATH = "models/gen"

#### Text Classification ####
TC_MODEL = "google-bert/bert-base-uncased"
# 2 labels (default)
TC_MODEL_PATH_2L = "models/tc/tc_2l/"
# 3 labels
TC_MODEL_PATH_3L = "models/tc/tc_3l/"

# With labels
TC_MODEL_LABELS = "cardiffnlp/tweet-topic-21-multi"
TC_MODEL_PATH_LABELS = "models/tc/tc_labels/"

# provide labels
TC_MODEL_PATH_PROVIDE_LABELS = "models/tc/provide_labels/"
PROVIDED_LABELS = ["PROVIDED_LABEL_0", "PROVIDED_LABEL_1", "PROVIDED_LABEL_2"]

#### Text-to-Text
TT_MODEL = "google/t5-efficient-tiny-el12"
TT_MODEL_PATH = "models/tt/"

### Sentence Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_PATH = "models/embedding/"

#### Cross Encoder Model
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSS_ENCODER_PATH = "models/cross_encoder/"


def get_chat_model():
    chat_model = EricChat(model_name=CHAT_MODEL)
    chat_model.save(CHAT_MODEL_PATH)


def get_chat_model_mlx():
    chat_model = EricChatMLX(model_name=CHAT_MODEL_MLX)
    chat_model.save(CHAT_MODEL_PATH_MLX)


def get_gen_model():
    gen_model = EricGeneration(model_name=GEN_MODEL)
    gen_model.save(GEN_MODEL_PATH)


def get_tc_model_2():
    tc_model_2 = EricTextClassification(model_name=TC_MODEL)
    tc_model_2.save(TC_MODEL_PATH_2L)


def get_tc_model_3():
    tc_model_3 = EricTextClassification(model_name=TC_MODEL, labels=["LABEL_0", "LABEL_1", "LABEL_2"])
    tc_model_3.save(TC_MODEL_PATH_3L)


def get_tc_with_labels():
    tc_with_labels = EricTextClassification(model_name=TC_MODEL_LABELS)
    tc_with_labels.save(TC_MODEL_PATH_LABELS)


def get_tc_with_provided_labels():
    tc_with_provided_labels = EricTextClassification(
        model_name=TC_MODEL, labels=PROVIDED_LABELS
    )
    tc_with_provided_labels.save(TC_MODEL_PATH_PROVIDE_LABELS)


def get_tt_model():
    tt_model = EricTextToText(model_name=TT_MODEL)
    tt_model.save(TT_MODEL_PATH)


def get_embedding_model():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_model.save(EMBEDDING_MODEL_PATH)


def get_cross_encoder_model():
    cross_encoder_model = CrossEncoder(model_name=CROSS_ENCODER_MODEL)
    cross_encoder_model.save(CROSS_ENCODER_PATH)


def main():
    print("get_chat_model()")
    get_chat_model()

    print("get_gen_model()")
    get_gen_model()

    print("get_tc_model_2()")
    get_tc_model_2()

    print("get_tc_model_3()")
    get_tc_model_3()

    print("get_tc_with_labels()")
    get_tc_with_labels()

    print("get_tc_with_provided_labels()")
    get_tc_with_provided_labels()

    print("get_tt_model()")
    get_tt_model()

    print("get_embedding_model()")
    get_embedding_model()

    print("get_cross_encoder_model()")
    get_cross_encoder_model()

    if sys.platform in ("darwin"):
        print("get_chat_model_mlx()")
        get_chat_model_mlx()


if __name__ == "__main__":
    main()
