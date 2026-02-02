## Initialization 

1. model_name (string, PreTrainedModel or None) ("bert-base-uncased"): Provide a string that contains a Hugging Face ID or a path to a model directory. You can also provide an already loaded model.

2. trust_remote_code (bool) (optional): Set to True to trust remote code. Only set to True for repositories you trust.

```python
from erictransformer import EricTextClassification

eric_tc = EricTextClassification(model_name="bert-base-uncased", trust_remote_code=False)
```

## Call

Arguments:

1. text (string):  Text for the model to classify 

2. args (```TCCallArgs```) (TCCallArgs()): Currently a placeholder, no custom functionality is provided.

```python

from erictransformer import EricTextClassification

eric_tc = EricTextClassification(model_name="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

result = eric_tc("Hello world")

print(result) 
```


