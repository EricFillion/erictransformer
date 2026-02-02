
## JSONL Format
Provide a JSONL file in the following format. Available roles include "system", "user" and "assistant". 

```
{"messages":[{"role":"system","content": "You are a helpful assistant."}, {"role":"user", "content":"Hi!"},{"role":"assistant", "content": "Hello!"}, {"role": "user","content": "What's 2+2?"}, {"role":"assistant","content": "4"}]}
{"messages":[{"role":"system","content": "You are a helpful assistant 1."}, {"role":"user", "content":"Hi!"},{"role":"assistant","content":" Hello!"}, {"role":"user","content": "What's 2+2?"}, {"role":"assistant","content": "4"}]}
```

## train()

inputs: 

1. train_path (string) (required): a path file to a train file or a directory that contains train files. 

2. eval_filepath (string) (optional): a path file to an eval file or a directory that contains eval files.  

3. args (CHATTrainArgs) (optional): a dataclass with the common arguments found [here](../train-args.md).

```python
import json
from erictransformer import EricChat, EricTrainArgs

eric_chat = EricChat(model_name="openai/gpt-oss-20b")
args = EricTrainArgs(out_dir="eric_transformer")

train_data = [{"messages": [{"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello!"},
                            {"role": "user", "content": "What's 2+2?"}, {"role": "assistant", "content": "4"}]},
              {"messages": [{"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"},
                            {"role": "user", "content": "What's 1+1?"}, {"role": "assistant", "content": "2"}]}
              ]

with open("data.jsonl", "w") as f:
    for td in train_data:
        f.write(json.dumps(td) + "\n")

result = eric_chat.train(train_path="data.jsonl", eval_path="data.jsonl", args=args)

print(result.final_train_loss)  # float
print(result.final_eval_loss)  # float
print(result.best_eval_loss)  # float
```

View the output directory in eric_transformer/

## eval()

Inputs:

1. train_path (string) (required): Same as train()'s train_path parameter. 

2. args (CHATEvalArgs) (optional): a dataclass with [these](../eval-args.md) arguments.

```python
import json
from erictransformer import EricChat, EricEvalArgs

eric_chat = EricChat(model_name="openai/gpt-oss-20b")
args = EricEvalArgs(out_dir="eric_transformer")

eval_data = [{"messages": [{"role": "system", "content": "You are a helpful assistant."},
                           {"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello!"},
                           {"role": "user", "content": "What's 2+2?"}, {"role": "assistant", "content": "4"}]},
             {"messages": [{"role": "system", "content": "You are a helpful assistant."},
                           {"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"},
                           {"role": "user", "content": "What's 1+1?"}, {"role": "assistant", "content": "2"}]}
             ]

with open("eval.jsonl", "w") as f:
    for td in eval_data:
        f.write(json.dumps(td) + "\n")

result = eric_chat.eval("eval.jsonl", args=args)

print("RESULT:", result.loss)  # float 
```

You can view the tokenized data in eric_transformer/
