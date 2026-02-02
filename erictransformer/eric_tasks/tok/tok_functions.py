from typing import Union

from datasets import Dataset
from transformers import PreTrainedTokenizer

from erictransformer.exceptions import EricTokenizationError


def get_max_in_len(
    max_len: Union[int, None], tokenizer: PreTrainedTokenizer
) -> int:
    if max_len >=1:
        return max_len
    else:
        if tokenizer.model_max_length > 10_000_000:
            return 512
        return tokenizer.model_max_length


def tokenize_gen(
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    max_len: int,
    bs: int,
    procs: int = 1,
) -> Dataset:
    def tokenize_fn(batch):
        try:
            tokens = tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )
            labels = []
            for input_ids in tokens["input_ids"]:
                masked = [
                    token if token != tokenizer.pad_token_id else -100
                    for token in input_ids
                ]
                labels.append(masked)
            tokens["labels"] = labels
            return tokens
        except Exception as e:
            raise EricTokenizationError(
                f"Tokenization failed during batch processing. Error: {e}"
            )

    try:
        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            batch_size=bs,
            desc="Tokenizing...",
            num_proc=procs,
        )
        cols = ["input_ids", "attention_mask"]
        cols.append("labels")

        tokenized.set_format("torch", columns=cols)
        return tokenized
    except Exception as e:
        raise EricTokenizationError(
            f"Failed during dataset mapping or formatting. Error: {e}"
        )


def tokenize_chat_template(
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    max_len: int,
    bs: int,
    procs: int = 1,
) -> Dataset:
    def tokenize_fn(example):
        try:
            inputs = [msg for msg in example["messages"]]
            tokens = tokenizer.apply_chat_template(
                inputs,
                tokenize=True,
                add_generation_prompt=False,
                max_length=max_len,
                padding="max_length",
                return_dict=True,
                truncation=True,
            )

            labels = []
            for input_ids in tokens["input_ids"]:
                masked = [
                    token if token != tokenizer.pad_token_id else -100
                    for token in input_ids
                ]
                labels.append(masked)
            tokens["labels"] = labels

            return tokens
        except Exception as e:
            raise EricTokenizationError(
                f"Tokenization failed during chat template application. Error: {e}"
            )

    try:
        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["messages"],
            batch_size=bs,
            desc="Tokenizing...",
            num_proc=procs,
        )
        tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return tokenized
    except Exception as e:
        raise EricTokenizationError(
            f"Failed during dataset mapping or formatting. Error: {e}"
        )
