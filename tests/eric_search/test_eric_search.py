import pathlib
import sys

import torch

from ericsearch import  (EricRanker,
    SearchCallArgs,
    EricSearch,
    SearchTrainArgs)


from erictransformer import CHATCallArgs, EricChat

if sys.platform in ("darwin"):
    from erictransformer import EricChatMLX

from tests.get_test_models import (
    CHAT_MODEL_PATH,
    CROSS_ENCODER_PATH,
    EMBEDDING_MODEL_PATH,
    CHAT_MODEL_PATH_MLX

)


def pad_or_trim(items: list[float], target_len: int):
    items = items[:target_len]
    items += [0] * (target_len - len(items))
    return items


vector_data = """
{ "text": "Ottawa is the capital of Canada", "metadata": {"id": "a" } }
{ "text": "Kingston was once the capital of Canada", "metadata": {"id": "b" }  }
{ "text": "Toronto is the largest city in Canada", "metadata": {"id": "c" }  }
""".strip()


class MockEmbeddingsModel:
    """Very naive embeddings model that runs super fast."""

    invert: bool

    def __init__(self) -> None:
        self.invert = False

    def encode(self, texts: list[str]) -> torch.Tensor:
        # Literally just number characters. Works in this tiny example.
        emb = torch.tensor(
            [pad_or_trim([float(ord(char)) for char in text], 10) for text in texts]
        )
        if self.invert:
            emb *= -1
        return emb

    def metadata(self):
        return "metadata"


def test_eric_search_chat(tmp_path: pathlib.Path):
    (tmp_path / "inputs").mkdir()

    with open(tmp_path / "inputs" / "input_data.jsonl", "w") as f:
        f.write(vector_data.strip())

    eric_ranker = EricRanker(model_name=CROSS_ENCODER_PATH)

    eric_search = EricSearch(model_name=EMBEDDING_MODEL_PATH, eric_ranker=eric_ranker)
    eric_search.eric_ranker.ignore_original_score = True
    eric_search.train(
        train_path=str((tmp_path / "inputs")),
        args=SearchTrainArgs(out_dir=str(tmp_path / "output_0")),
    )
    eric_chat = EricChat(model_name=CHAT_MODEL_PATH, eric_search=eric_search)

    hd_args = SearchCallArgs(
        limit=4, leaf_count=4, ranker_count=2, bs=2
    )
    args = CHATCallArgs(search_args=hd_args)
    result = eric_chat("tell me about ottawa", args=args)

    assert type(result.text) == str

    for i, r in enumerate(eric_chat.stream("tell me about ottawa")):
        if i == 0:
            assert r.marker == "search"
        elif i == 1:
            assert r.marker == "search_result"
            assert r.payload["best_sentence"]
            assert r.payload["metadata"]



def test_eric_search_chat_mlx(tmp_path: pathlib.Path):
    if sys.platform in ("darwin"):
        (tmp_path / "inputs").mkdir()

        with open(tmp_path / "inputs" / "input_data.jsonl", "w") as f:
            f.write(vector_data.strip())

        eric_ranker = EricRanker(model_name=CROSS_ENCODER_PATH)

        eric_search = EricSearch(model_name=EMBEDDING_MODEL_PATH, eric_ranker=eric_ranker)
        eric_search.eric_ranker.ignore_original_score = True
        eric_search.train(
            train_path=str((tmp_path / "inputs")),
            args=SearchTrainArgs(out_dir=str(tmp_path / "output_0")),
        )
        eric_chat = EricChatMLX(model_name=CHAT_MODEL_PATH_MLX, eric_search=eric_search)

        hd_args = SearchCallArgs(
            limit=4, leaf_count=4, ranker_count=2, bs=2
        )
        eric_search = EricSearch(model_name=CHAT_MODEL_PATH)
        args = CHATCallArgs(search_args=hd_args)
        result = eric_chat("tell me about ottawa", args=args)

        for i, r in enumerate(eric_chat.stream("tell me about ottawa")):
            if i == 0:
                assert r.marker == "search"
            elif i == 1:
                assert r.marker == "search_result"
                assert r.payload["best_sentence"]
                assert r.payload["metadata"]
