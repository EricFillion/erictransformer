from typing import List, Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from erictransformer.exceptions import EricLoadModelError, EricLoadTokenizerError


def _get_torch_dtype(torch_dtype: str) -> Union[torch.dtype, str]:
    if torch_dtype == "fp32":
        return torch.float32
    elif torch_dtype == "fp16":
        return torch.float16
    elif torch_dtype == "bf16":
        return torch.bfloat16
    else:
        raise ValueError(
            f"Invalid torch_dtype {torch_dtype}. Provide one of auto, fp32, fp16, or bf16."
        )


def get_model(
    model_name_path: Union[str, PreTrainedModel],
    model_class: AutoModel,
    trust_remote_code: bool,
    precision,
) -> PreTrainedModel:
    try:
        if isinstance(model_name_path, PreTrainedModel):
            return model_name_path

        loaded_config = AutoConfig.from_pretrained(
            model_name_path, trust_remote_code=trust_remote_code
        )
        torch_dtype = _get_torch_dtype(precision)

        model = model_class.from_pretrained(
            model_name_path,
            config=loaded_config,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
        )

        return model

    except Exception as e:
        raise EricLoadModelError(f"Failed to load model from '{model_name_path}': {e}")


def et_retrieve_tokenizer(
    tokenizer_path: str, trust_remote_code: bool
):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
        )

        return tokenizer
    except Exception as e:
        raise EricLoadTokenizerError(
            f"Failed to load tokenizer from '{tokenizer_path}': {e}"
        )


def get_tokenizer(
    model_name_path: str,
    tokenizer_path: Union[str, PreTrainedTokenizerBase],
    trust_remote_code: bool,
) -> PreTrainedTokenizerBase:
    if (
        tokenizer_path is None
    ):  # by default we use the same value provided to model_name_path
        tokenizer = et_retrieve_tokenizer(
            tokenizer_path=model_name_path,
            trust_remote_code=trust_remote_code,
        )

    elif isinstance(tokenizer_path, str):
        tokenizer = et_retrieve_tokenizer(
            tokenizer_path=tokenizer_path,
            trust_remote_code=trust_remote_code,
        )

    elif isinstance(tokenizer_path, PreTrainedTokenizerBase):
        # tokenizer provided directly
        tokenizer = tokenizer_path
        # maybe remove the following code since if an advanced user provides their own tokenizer they might want full control
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(
            "Invalid tokenizer_path. Provide a None, a string or AutoTokenizer"
        )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    return tokenizer


def get_model_components(
    model_name_path: Union[str, PreTrainedModel, None],
    trust_remote_code: bool,
    model_class: AutoModel,
    tokenizer_path: Union[str, PreTrainedTokenizerBase],
    precision: str,
):
    tokenizer = get_tokenizer(
        model_name_path=model_name_path,
        tokenizer_path=tokenizer_path,
        trust_remote_code=trust_remote_code,
    )

    if model_name_path is not None:
        model = get_model(
            model_name_path=model_name_path,
            model_class=model_class,
            trust_remote_code=trust_remote_code,
            precision=precision,
        )
        config = model.config
    else:
        model = None
        config = None

    return config, tokenizer, model


def get_model_components_tc(
    model_name_path: Union[str, PreTrainedModel, None],
    trust_remote_code: bool,
    model_class: AutoModel,
    tokenizer_path: Union[str, PreTrainedTokenizerBase],
    precision: str,
    labels: Optional[List[str]] = None,
):

    config = AutoConfig.from_pretrained(
        model_name_path, trust_remote_code=trust_remote_code
    )
    reset_labels =  False  # set to True if we should reset the labels to the size of num_labels

    config_id2label = getattr(config, "id2label", None)
    config_label2id = getattr(config, "label2id", None)


    if not config_id2label or not config_label2id:
        reset_labels = True
    elif labels is not None:
        reset_labels = True
    else:
        id2label_length = len(config_id2label)
        label2id_length = len(config_label2id)
        if id2label_length != label2id_length:
            reset_labels = True


    if reset_labels:
        if labels is None:
            labels = ["LABEL_0", "LABEL_1"]

        if labels:
            config.id2label = {i: labels[i] for i in range(len(labels))}
        else:
            num_labels = 2
            config.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}

        config.label2id = {v: k for k, v in config.id2label.items()}

    config.num_labels = len(config.id2label)

    torch_dtype = _get_torch_dtype(precision)

    model = model_class.from_pretrained(
        model_name_path,
        config=config,
        dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        ignore_mismatched_sizes=True,
    )

    tokenizer = get_tokenizer(
        model_name_path=model_name_path,
        tokenizer_path=tokenizer_path,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    return config, tokenizer, model
