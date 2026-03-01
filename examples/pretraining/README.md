# Pretraining Instructions

Examples: 

Generation: [generation/common_corpus 
](./generation/common_corpus)

Text-to-text (translation): [text_to_text/translation](./text_to_text/translation)

Some examples have their own unique arguments in addition to the ones below which you can find in the example's README.md.

The weights for the model you use will be reset allowing to allow you to retrain it from scratch. 


## Basic Usage

### data.py
```commandline
python3 data.py --train_cases 131_072 --eval_cases 64 
```
train_cases (int): The number of cases from the dataset that will be used to produce training cases.

eval_cases (int): The number of cases from the dataset that will be used to produce eval cases. 

_Note: train_cases/eval_cases represents the number of "raw" cases which are then processed into the final train/eval cases. So, the final number of train/eval cases could be different than what's provided._ 


### tok.py
```commandline
python3 tok.py --model_name {MODEL-NAME}
```
model_name (str): Path/name for a tokenizer

###  train.py
```commandline
python3 train.py --model_name {MODEL-NAME} --bs 1 --eval_steps 256 --log_steps 256 --checkpoint_steps -1 --lr 1e-5
```
model_name (str): The name/path for the model that will be loaded to copy its structure for the starting model. Its weights will not be used. 

bs (int): The training batch size. Increasing it speeds up training but increases memory usage. 

eval_steps (int): How many steps until eval. 

log_steps (int): How many steps until logging. 

checkpoint_steps (int): How many steps until checkpointing. When -1 no checkpointing is performed. 

lr (float): The learning rate. 


### sample.py
```commandline
python3 sample.py --model_name model --text "Hello world"
```

or 

```commandline
python3 sample.py --checkpoint --text "Hello world"
```

model_name (str): Path to a dir or  hf model id for a text generation model. By default, train.py saves the resulting trained model to the "model/" dir.

checkpoint (bool): When provided, the most recent training output directory under the eric_transformer folder will be inspected, and the latest checkpoint will be used.

text (str): The starting text for the model. 


### push.py
First log into Hugging Face. 

```commandline
 hf auth login 
```

```commandline
python3 push.py --model model --hf_id {REPO-ID} 
```
model: The model 

hf_id: Hugging Face model ID. 

If you would like to make the model public, then go into the repo's settings on Hugging Face. 


## Advanced Usage

### Private models 
To use private models log into Hugging Face and provide ```--use_auth_token``` to ```tok.py``` and ```train.py```
```commandline
 hf auth login 
```

### data.py Advanced 
```commandline
python3 data.py --train_cases 1024 --eval_cases 64   --untok_train data/untok/train/ --untok_eval data/untok/eval/ --shards 8  --use_auth_token 
```
untok_train (str): The path where the train data will be stored 

untok_eval (str): The path where the eval data will be stored 

shards (int): How many files the train and eval data are broken into

seed (int): Controls randomness 

### tok.py (advanced)
```commandline
python3 tok.py --untok_train data/untok/train/ --untok_eval data/untok/eval/  --tok_train data/tok/train/ --tok_eval data/tok/eval/  --shards 8  --use_auth_token  --seed 42
```
untok_train (str): Same path you used for data.py

untok_eval (str): Same path you used for data.py

tok_train (str): The path where train tokenized data will be saved 

tok_eval (str): The path where eval tokenized data will be saved 

shards (int): The number of files the tokenized cases are spread out into 

use_auth_token (bool): When provided private Hugging Face models can be accessed using the local HF key. 

seed (int): Controls randomness 


### tok.py (fast)

Note: If you experience problems it could be from multiprocessing so disable --procs by setting it to -1. 
```commandline
python3 tok.py  --procs 0 --bs 2048
```
procs(int):  When -1 multiprocessing is disabled. When 0 half the CPU's cores are used. When >0 it indicates the number of cores to be used.  

bs (int): The batch size for tokenizing using Hugging Face's Dataset.map() functionality. 


### train.py (advanced)

```commandline
python3 train.py  --eval_bs 0 --epochs 1  --gas 1 --optim adamw --save_model_path model --out_dir eric_transformer --run_name run_1 --seed 42
```

eval_bs (int): Eval batch size. When 0 (the default) the train batch size is used. You can safely set eval_bs higher than bs without running out of memory. 

epochs (int): The number of train epochs. 

gas (int): The number of gradient accumulation steps. 

optim (str): Either "adamw" or "sgd" to specify the optim. AdamW typically performs better than SGD but requires more memory. For training super large models, like GPT-OSS-20b, SGD should be used.

save_best (bool): When provided the model with the lowest loss is saved (If enabled we recommend increasing eval_steps so not too much time is spent saving)

save_model_path (str): Path where the final model is saved 

out_dir (str): A folder that will be created to store your runs 

run_name (str): When empty, a new folder with a timestamp is made for the run. When a string is provided that name is used. 

resume_path (str): A path to a checkpoint folder to resume a run 

seed (int): Controls randomness 


