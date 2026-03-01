# Fine-tune a text generation model for instruction following 


Please refer to the general fine-tuning [instructions](../../README.md). 


## Sample.py

context (str): A string that's used to provide context for RAG

```commandline
python3 sample.py  --context "Eric Transformer is a single open-source Python codebase for AI that supports pretraining, fine-tuning and retrieval augmented generation (RAG). Many of its components are built with just pure PyTorch making it lightweight and highly optimized." --text "tell me about Eric Transformer"
```

bg_text: text that contains background information. 
