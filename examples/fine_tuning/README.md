# Fine-tuning Instructions 

Examples: 

Chat: [chat/databricks/ 
](./chat/databricks)

Generation: [generation/billsum/ 
](./generation/billsum)

Text classification (emotion classification): [text_classification/go_emotion/ 
](./text_classification/go_emotion)


Some of the examples have their own unique arguments in addition to the ones below which you can find in the example's README.md.


##  train.py
```commandline
python3 train.py --model_name {MODEL-NAME} --train_cases 1024 --eval_cases 256 --bs 1 --eval_steps 256 --log_steps 256 --checkpoint_steps -1 --lr 1e-5
```
train_cases (int): The number of cases from the dataset that will be used to produce training cases.

eval_cases (int):The number of cases from the dataset that will be used to produce eval cases.

model_name (str): The name/path for the model that will be loaded to copy its structure for the starting model. Its weights will not be used. 

bs (int): The training batch size. Increasing it speeds up training but increases memory usage. 

eval_steps (int): How many steps until eval. 

log_steps (int): How many steps until logging. 

checkpoint_steps (int): How many steps until checkpointing. When -1 checkpointing is disabled is done

lr (float): The learning rate. 

_Note: train_cases/eval_cases represents the number of "raw" cases which are then processed into the final train/eval cases. So, the final number of train/eval cases could be different than what's provided._ 


## sample.py

text (str): text that's provided to the model 

model_name (str): A path to the model or a Hugging Face ID. By default it's a path to where the model from train.py is saved after running. 

checkpoint (bool): When provided, the most recent checkpoint is used for the model. Takes priority over the 'model_name' argument.


```commandline
python3 sample.py --text "Ottawa." 
```

## train.py (advanced)

```commandline
python3 train.py  --eval_bs 0 --epochs 1  --gas 1 --optim adamw --train_path data/train.jsonl --eval_path data/eval.jsonl  --save_model_path model --out_dir eric_transformer --run_name run_1 --seed 42
```

eval_bs (int): Eval batch size. When 0 (the default) the train batch size is used. You can safely set eval_bs higher than bs without running out of memory. 

epochs (int): The number of train epochs. 

gas (int): The number of gradient accumulation steps. 

optim (str): Either "adamw" or "sgd" to specify the optim. AdamW typically performs better than SGD but requires more memory. For training super large models, like GPT-OSS-20b, SGD should be used. 

save_best (bool): When provided the model with the lowest loss is saved (If enabled we recommend increasing eval_steps so not too much time is spent saving)

train_path (str): A path to a JSONL file for where the train data is saved. 

eval_path: (str): A path to a JSONL file for where the eval data is saved. 

save_model_path (str): Path where the final model is saved 

out_dir (str): A folder that will be created to store your runs 

run_name (str): When empty, a new folder with a timestamp is made for the run. When a string is provided that name is used. 

resume_path (str): A path to a checkpoint folder to resume a run 

seed (int): Controls randomness 





