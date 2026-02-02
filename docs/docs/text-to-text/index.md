## Initialization 

1. model_name (string, PreTrainedModel or None) ("google-t5/t5-base"):  Provide a string that contains a Hugging Face ID or a path to a model directory. You can also provide an already loaded model. 

2. trust_remote_code (bool) (optional): Set to True to trust remote code. Only set to True for repositories you trust.

```python
from erictransformer import EricTextToText

# You can use a larger model if you have enough memory. The below page shows models between 60m to 11b parameters. 
# https://huggingface.co/google-t5

eric_gen = EricTextToText(model_name="google-t5/t5-base", trust_remote_code=False)
```

## Call

Arguments:

1. text (string): The text prompt for the model.

2. args (```TTCallArgs```) (TTCallArgs()): See this [webpage](../call-args.md) for more detail.

```python

from erictransformer import EricTextToText, TTCallArgs

eric_tt = EricTextToText(model_name="google-t5/t5-small")

args = TTCallArgs(  # Min/max number of tokens to produce during generation.
    min_len=1,
    max_len=32,  # short translations 
    # Sampling settings. 
    temp=0.6, 
    top_k=32, 
    top_p=0.8  
)

result = eric_tt("Translate English to French: Hello how are you?", args=args)

print(result.text)  # str 
```

