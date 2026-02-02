## Initialization

1. model_name (string, PreTrainedModel or None) ("openai/gpt-oss-20b"): Provide a string that contains a Hugging Face ID or a path to a model directory. You can also provide an already loaded model.

2. trust_remote_code (bool) (optional): Set to True to trust remote code. Only set to True for repositories you trust.

```python
from erictransformer import EricChat

eric_chat = EricChat(model_name="openai/gpt-oss-20b", trust_remote_code=False)
```

## Recommended Models 

We recommended you use the following model, with an H200 or B200 GPU, as we support its chat template for both inference and fine-tuning. 

[openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)

We also provide have experimental support Hugging Face's [SmolLM3-3B](HuggingFaceTB/SmolLM3-3B) model. 

## Call

Arguments:

1. text (string or List[dict]): Either an input user prompt or a list of dictionaries that contain a conversation history 

2. args (```CHATCallArgs```) (CHATCallArgs()): See this [webpage](../call-args.md) for more detail.

```python

from erictransformer import EricChat, CHATCallArgs

eric_chat = EricChat(model_name="openai/gpt-oss-20b")

args = CHATCallArgs(  # Min/max number of tokens to produce during generation.
    min_len=1,
    max_len=1024,
    # Sampling settings. 
    temp=1.0,
    top_k=50,
    top_p=1.0)

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
from erictransformer import EricChat, CHATCallArgs

eric_chat = EricChat(model_name="openai/gpt-oss-20b")

args = CHATCallArgs(  # Min/max number of tokens to produce during generation.
    min_len=1,
    max_len=1024,
    # Sampling settings. 
    temp=1.0,
    top_k=50,
    top_p=1.0)

text = "Ottawa Canada."  # you can instead provide a conversation like in the previous example 

for token in eric_chat.stream(text, args=args):
    print(token.text, end="")

```

## RAG: EricChat() + EricSearch()

See Eric Search's GitHub [repo](https://github.com/EricFillion/ericsearch) for more info. 

You can provide a EricSearch() object to EricChat() to utilize rag while performing inference. You can also adjust the following parameters for SearchCallArgs within CHATCallArgs: 

1. leaf_count (int) (32): The number of leaves that we search. Increase it to search for more relevant documents, at the cost of a slower search time. 

2. ranker_count (int) (4):  The number of documents that are sent to EricRanker() for information extraction. Increasing this greatly decreases speed but improves accuracy. 

3. bs (int) (32): Batch size. 

```python
from ericsearch import EricSearch, SearchCallArgs
from erictransformer import EricChat, CHATCallArgs

eric_search = EricSearch(data_name="EricFillion/ericsearch-hello-world")
eric_chat = EricChat(model_name="openai/gpt-oss-20b", eric_search=eric_search)

args = CHATCallArgs(
    search_args=SearchCallArgs(leaf_count=32, ranker_count=4, bs=32),
    min_len=1, max_len=1024, temp=1.0, top_k=50, top_p=1.0)

result = eric_chat("Hello world", args=args)

print(result.text)

for chunk in eric_chat.stream("Hello world", args=args):
    print(chunk.text, end="")

    # If you would like more detail, such as the result of the RAG search, you see chunk's other arguments 
    # print(chunk)

```


