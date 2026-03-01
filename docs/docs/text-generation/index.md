## Initialization 

1. model_name (string, PreTrainedModel or None) ("cerebras/Cerebras-GPT-111M"): Provide a string that contains a Hugging Face ID or a path to a model directory. You can also provide an already loaded model.  

2. trust_remote_code (bool) (optional): Set to True to trust remote code. Only set to True for repositories you trust.

```python
from erictransformer import EricGeneration

# we recommend you use a larger model if you have enough memory. Cerebras has models ranging from 111M to 13B parameters. 
# https://huggingface.co/collections/cerebras/cerebras-gpt

eric_gen = EricGeneration(model_name="cerebras/Cerebras-GPT-111M", trust_remote_code=False)
```

## Call

Arguments: 

1. text (string): The text prompt for the model.

2. args (```GENCallArgs```) (GENCallArgs()): See this [webpage](../call-args.md) for more detail.

```python

from erictransformer import EricGeneration, GENCallArgs

eric_gen = EricGeneration(model_name="cerebras/Cerebras-GPT-111M")

args = GENCallArgs(  # Min/max number of tokens to produce during generation.
    min_len=1,
    max_len=1024,
    # Sampling settings. 
    temp=0.6,
    top_k=32,
    top_p=0.8)

result = eric_gen("Hello world", args=args)

print(result.text)  # str 
```

