All Eric Transformer classes (EricGeneration, EricChat, EricTextClassification and EricTextToText) can be pushed to Hugging Face's [Model Hub](https://huggingface.co/).

First log into Hugging Face 

```bash
hf auth login
```

## Push

The _push()_ method contains 2 arguments:

1. repo_name (string) (required): The name of the Hugging Face ID

2. private (bool) (True): If True, a private repo is used/created. If False, a public one is created.

```python
from erictransformer import EricGeneration

eric_gen_1 = EricGeneration(model_name="cerebras/Cerebras-GPT-111M")
repo_name = "{REPO ID}"
eric_gen_1.push(repo_name, private=True)

eric_gen_2 = EricGeneration(model_name=repo_name)
```



