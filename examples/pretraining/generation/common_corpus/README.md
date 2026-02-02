# Pretraining Generation  

Please refer to the general pretraining [instructions](../../README.md). 


Pretrain a model with a dataset called [PleIAs/common_corpus](https://huggingface.co/datasets/PleIAs/common_corpus). 

## data.py

```commandline
python3 data.py --train_cases 1024 --eval_cases 64  --datasets  OpenScience OpenWeb OpenCulture OpenGovernment  OpenSemantic
```

--datasets: Provide a list of strings that contains sub-datasets to use. The options include: OpenScience, OpenWeb, OpenCulture, OpenSource, OpenGovernment and or OpenSemantic


