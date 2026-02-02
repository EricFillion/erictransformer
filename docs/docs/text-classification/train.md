## JSONL Format

The text field contains the text sample for the training case. 
The label fields must be an int to represent what ID the text belongs to.

```
{"text": "Sample train case ", "label": 0} 
{"text": "Another sample train case ", "label": 1} 
```

## Labels IDs

You can see what labels and IDs are available for your model with the following code

```python
from erictransformer import EricTextClassification

eric_tc_labels = EricTextClassification(model_name="bert-base-uncased", labels=["custom_0", "custom_1", "custom_3"])

print(eric_tc_labels.config.id2label)  # {0: 'custom_0', 1: 'custom_1', 2: 'custom_3'}
print(eric_tc_labels.config.label2id)  # {'custom_0': 0, 'custom_1': 1, 'custom_3': 2}

# when fine-tuning provide the ID 0 for custom_0, 1 for 'custom_1' and 2 for 'custom_3'


```

### train()

inputs: 
1. train_path (string) (required): a path file to a train file or a directory that contains train files. 

2. eval_filepath (string) (optional): a path file to an eval file or a directory that contains eval files.  

3. args (EricTrainArgs) (optional): a dataclass with the  arguments found [here](../train-args.md).

```python
import json
from erictransformer import EricTextClassification, EricTrainArgs

eric_gen = EricTextClassification(model_name="bert-base-uncased", labels=["LABEL_0", "LABEL_1", "LABEL_3"])
args = EricTrainArgs(out_dir="eric_transformer", lr=2e-5)
train_data = [{"text": "Train data 0", "label": 0},
              {"text": "Train data 1", "label": 1},
              {"text": "Train data 2", "label": 2}
              ]

with open("data.jsonl", "w") as f:
    for td in train_data:
        f.write(json.dumps(td) + "\n")

result = eric_gen.train(train_path="data.jsonl", eval_path="data.jsonl", args=args)

print(result)
```

View the output directory in eric_transformer/


### eval()

Inputs:

1. train_path (string) (required): Same as train()'s train_path parameter. 

2. args (EricEvalArgs) (optional): a dataclass with [these](../eval-args.md) arguments.

```python
import json
from erictransformer import EricTextClassification, EricEvalArgs

eric_tc = EricTextClassification(model_name="bert-base-uncased")
args = EricEvalArgs(out_dir="eric_transformer")

train_data = [{"text": "Eval data 0", "label": 0}, {"text": "Eval data 1", "label": 1},
              ]
with open("eval.jsonl", "w") as f:
    for td in train_data:
        f.write(json.dumps(td) + "\n")

result = eric_tc.eval("eval.jsonl", args=args)

print(result.loss) 
```

You can view the tokenized data in eric_transformer/
