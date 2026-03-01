## JSONL Format

```
{"text": "Line 1: This file contains text to train a text generation model"} 
{"text": "Line 2: This file contains text to train a text generation model"}
```

## train()

Inputs: 

1. train_path (string) (required): a path file to a train file or a directory that contains train files. 

2. eval_filepath (string) (optional): a path file to an eval file or a directory that contains eval files.  

3. args (TrainArgs) (optional): a dataclass with the common arguments found [here](../train-args.md).

```python
import json
from erictransformer import EricGeneration, EricTrainArgs

eric_gen = EricGeneration(model_name="cerebras/Cerebras-GPT-111M")
args = EricTrainArgs(out_dir="eric_transformer")
train_data = [{"text": "Train data 1"}, {"text": "Train data 2"}]
with open("data.jsonl", "w") as f:
    for td in train_data:
        f.write(json.dumps(td) + "\n")

result = eric_gen.train(train_path="data.jsonl", eval_path="data.jsonl", args=args)

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
from erictransformer import EricGeneration, EricEvalArgs

eric_gen = EricGeneration(model_name="cerebras/Cerebras-GPT-111M")
args = EricEvalArgs(out_dir="eric_transformer", run_name="gen_eval_example")

train_data = [{"text": "eval data 1"}, {"text": "eval data 2"}]
with open("eval.jsonl", "w") as f:
    for td in train_data:
        f.write(json.dumps(td) + "\n")

result = eric_gen.eval("eval.jsonl", args=args)

print("RESULT:", result.loss)  # float 
```

You can view the tokenized data in eric_transformer/gen_eval_example/tok_eval_data
