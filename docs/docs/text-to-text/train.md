## JSONL Format

```
{"input": "example input text 1 ", "target": "example output text 1"} 
{"input": "example input text 2", "target": "example output text 2"} 
```

## train()

inputs: 

1. train_path (string) (required): a path file to a train file or a directory that contains train files. 

2. eval_filepath (string) (optional): a path file to an eval file or a directory that contains eval files.  

3. args (TrainArgs) (optional): a dataclass with the arguments found [here](../train-args.md).

```python
import json
from erictransformer import EricTextToText, EricTrainArgs

eric_tt = EricTextToText(model_name="google-t5/t5-base")

args = EricTrainArgs(out_dir="eric_transformer")

train_data = [{"input": "example input text 1", "target": "example output text 1"},
              {"input": "example input text 2", "target": "example output text 2"}
              ]

with open("data.jsonl", "w") as f:
    for td in train_data:
        f.write(json.dumps(td) + "\n")

result = eric_tt.train(train_path="data.jsonl", eval_path="data.jsonl", args=args)

print(result.final_train_loss)  # float
print(result.final_eval_loss)  # float
print(result.best_eval_loss)  # float
```

View the output directory in eric_transformer/

## eval()

Inputs:
1. train_path (string) (required): Same as train()'s train_path parameter. 

2. args (EvalArgs) (optional): a dataclass with [these](../eval-args.md) arguments.

```python

import json

from erictransformer import EricTextToText, EricEvalArgs

eric_tt = EricTextToText(model_name="google-t5/t5-base")
args = EricEvalArgs(out_dir="eric_transformer")

eval_data = [{"input": "example input text 1", "target": "example output text 1"},
             {"input": "example input text 2", "target": "example output text 2"}
             ]

with open("eval.jsonl", "w") as f:
    for ed in eval_data:
        f.write(json.dumps(ed) + "\n")

result = eric_tt.eval("eval.jsonl", args=args)

print("RESULT:", result.loss)  # float 
```

You can view the tokenized data in eric_transformer/
