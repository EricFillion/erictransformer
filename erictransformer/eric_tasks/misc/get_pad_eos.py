from transformers import PreTrainedModel, PreTrainedTokenizer

from erictransformer.exceptions import EricInferenceError


def get_pad_eos(tokenizer: PreTrainedTokenizer, model: PreTrainedModel):
    if tokenizer.pad_token_id is not None:
        pad_id = tokenizer.pad_token_id
    elif tokenizer.eos_token_id is not None:
        pad_id = tokenizer.eos_token_id
    else:
        raise EricInferenceError(
            "Tokenizer doesn't have a pad_token_id or eos_token_id token"
        )

    if model.config.eos_token_id is not None:
        eos_id = model.config.eos_token_id
    elif tokenizer.eos_token_id is not None:
        eos_id = tokenizer.eos_token_id
    else:
        raise EricInferenceError(
            "The model and the tokenizer don't define an eos_token_id"
        )
    return pad_id, eos_id
