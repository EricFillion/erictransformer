We recommend you use the "tok()" method to tokenize your data before training for large datasets. This provides greater control over tokenization and allows you to reuse tokenized data rather than retokenizing it for each training run. 

## TokArgs()

Args:

1.  max_cases (int) (-1): Maximum number of cases. When -1 no limit is set 

2.  shards (int) (1): Number of shard files to produce. Increasing this reduces memory usage which reduces memory usage. 

3.  bs (int) (1024): The batch size for tokenizing using Hugging Face's Dataset.map() functionality. 

4.  procs (int) (-1):  Number of processes to use. When -1, multiprocessing is disabled. When 0, resolves to half the CPU core count. When >0 it indicates the number of processes to be used.


```python
from erictransformer import GENTokArgs, CHATTokArgs, TCTokArgs

gen_tok_args = GENTokArgs(max_cases=1024, shards =2, bs=512, procs=0)
```

### Custom Parameters: GENTokArgs(), ChatTokArgs(), TCTokArgs()

```GENTokArgs```, ```CHATTokArgs``` and ```TCTokArgs``` all  have a custom parameter called ```max_len```  which trims the number of tokens for each case. 

```python
from erictransformer import GENTokArgs, CHATTokArgs, TCTokArgs

gen_tok_args = GENTokArgs(max_len=512)

chat_tok_args = CHATTokArgs(max_len=512)

tc_tok_args = TCTokArgs(max_len=512)
```

### Custom Parameters: TTTokArgs()

```TTTokArgs``` have separate parameters for trimming cases: 

1. max_in_len (int): Max token length for the input text 

2. max_in_len (int):  Max token length for the target text

```python
from erictransformer import TTTokArgs

tt_tok_args = TTTokArgs(max_in_len=512, max_out_len=512)
```

## Example

```python
import json
from erictransformer import EricGeneration, GENTokArgs

eric_gen = EricGeneration(model_name="cerebras/Cerebras-GPT-111M")

train_data = [{"text": "train data 1"}, {"text": "train data 2"}]

out_dir_train = "out_dir_train/"

with open("train.jsonl", "w") as f:
    for td in train_data:
        f.write(json.dumps(td) + "\n")

eval_data = [{"text": "eval data 1"}, {"text": "eval data 2"}]

out_dir_eval = "out_dir_eval/"

with open("eval.jsonl", "w") as f:
    for td in eval_data:
        f.write(json.dumps(td) + "\n")

args = GENTokArgs(max_cases=1024, shards=8, bs=4, procs=0)

eric_gen.tok("train.jsonl", out_dir=out_dir_train, args=args)

eric_gen.tok("eval.jsonl", out_dir=out_dir_eval, args=args)

train_result = eric_gen.train(train_path=out_dir_train, eval_path=out_dir_eval)

eval_result = eric_gen.eval(eval_path=out_dir_eval)
```
