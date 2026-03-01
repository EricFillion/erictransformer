Shared call args for ```EricTextGeneration()```, ```EricChat()``` and ```EricTextToText()``` 

```python
from erictransformer import GENCallArgs, CHATCallArgs, TTCallArgs
```

| Parameter    | Default | 
|--------------|---------|
| min_len      | 1       | 
| temp         | 0.8     | 
| top_k        | 32      | 
| top_p        | 0.6     | 


 ```max_len``` value based on CallArgs type.

| Parameter       | Default | 
|-----------------|---------|
| CHATCallArgs    | 4096    |
| GENCallArgs     | 1024    |
| TTCallArgs      | 1024    | 
