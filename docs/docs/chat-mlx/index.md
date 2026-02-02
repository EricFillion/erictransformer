
## Initialization

1. model_name (string) ("mlx-community/SmolLM3-3B-4bit").  Provide a string that contains a Hugging Face ID or a path to a model directory. We recommend you provide a model that has already converted to MLX.

```python
from erictransformer import EricChatMLX

eric_gen = EricChatMLX(model_name="mlx-community/SmolLM3-3B-4bit")
```

We recommended you use either an OpenAI GPT-OSS model or a Hugging Face SmolLM3-3B as we support their chat templates.

## Call

Arguments:

1. text (string or List[dict]): Either an input user prompt or a list of dictionaries that contain a conversation history 

2. args (```CHATCallArgs```) (CHATCallArgs()): See this [webpage](../call-args.md) for more detail.

```python

from erictransformer import EricChatMLX, CHATCallArgs

eric_chat = EricChatMLX(model_name="mlx-community/SmolLM3-3B-4bit")

args = CHATCallArgs(  # Min/max number of tokens to produce during generation.
    min_len=1,
    max_len=1024,
    # Sampling settings. 
    temp=1.0,
    top_k=32,
    top_p=0.6)

result = eric_chat("Hello world", args=args)

print(result.text)  # str 

# Conversation 
convo = [{"role": "system", "content": "You are a friendly ai assistant"},
         {"role": "user", "content": "what's 1 +1?"},
         {"role": "assistant", "content": "2"},
         {"role": "user", "content": "what's 2 +2?"}
         ]

result = eric_chat(convo, args=args)

print(result)  # str 


```

## stream()

The stream() method contains 2 arguments:

1. text (string or List[dict]): Either an input user prompt or a list of dictionaries that contain a conversation history.

2. args (```CHATCallArgs```) (CHATCallArgs()): See this [webpage](../call-args.md) for more detail.

```python

from erictransformer import EricChatMLX, CHATCallArgs

eric_chat = EricChatMLX(model_name="mlx-community/SmolLM3-3B-4bit")

args = CHATCallArgs(  # Min/max number of tokens to produce during generation.
    min_len=1,
    max_len=1024,
    # Sampling settings.
    temp=1.0,
    top_k=50,
    top_p=1.0)

text = "Hello world"  # you can instead provide a conversation like in the previous example

for token in eric_chat.stream(text, args=args):
    if token.marker == "text":
        print(token.text, end="")
    # you can also see the thinking tokens with token.marker == "thinking"

```


## RAG: EricSearch() + EricChatMLX()

EricSearch can be integrated with the EricChatMLX in the same way its be integrated with EricChat. See the bottom of this [page](../chat/index.md).
