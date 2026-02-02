All EricTransformer classes contain a method called ```save()``` that allow models to be saved locally. 
  
## save():
The method used to save models. It contains a single argument.

1. path: a file path to a directory to save various files. 
    Any previous files of the same names as created files will be overwritten. 
    We recommend that you use an empty directory.

```python
from erictransformer import EricGeneration

eric_gen = EricGeneration(model_name="cerebras/Cerebras-GPT-111M")
eric_gen.save(path="model/")
```

When initializing a Eric Transformer object, provide a path for the model_name parameter.

```python
from erictransformer import EricGeneration

# to use a private model first login Hugging Face with "hf auth login"

eric_gen = EricGeneration(model_name="model/")
```
