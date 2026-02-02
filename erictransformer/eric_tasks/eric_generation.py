import textwrap
import threading
from typing import List, Optional, Tuple, Union

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
    default_data_collator,
)

from erictransformer.args import EricTrainArgs, EricEvalArgs
from erictransformer.eval_models import EvalModel
from erictransformer.exceptions import EricInferenceError
from erictransformer.eric_tasks.args import (
    GENCallArgs,
    GENTokArgs,
)
from erictransformer.eric_tasks.misc import generate_gen_kwargs, get_pad_eos
from erictransformer.eric_tasks.results import GENResult
from erictransformer.eric_tasks.tok.tok_functions import (
    get_max_in_len,
    tokenize_gen,
)
from erictransformer.eric_transformer import EricTransformer, EricTransformerArgs
from erictransformer.loops import EvalResult
from erictransformer.utils import get_model_components
from erictransformer.validator import GENValidator


class EricGeneration(EricTransformer):
    def __init__(
        self,
        model_name: Union[str, PreTrainedModel, None] = "cerebras/Cerebras-GPT-111M",
        *,
        trust_remote_code: bool = False,
        tokenizer: Union[str, PreTrainedTokenizerBase] = None,
    ):
        model_class = AutoModelForCausalLM

        eric_args = EricTransformerArgs(
            model_name=model_name,
            model_class=model_class,
            trust_remote_code=trust_remote_code,
            tokenizer=tokenizer
        )

        super().__init__(eric_args)
        self.task_validator = GENValidator(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            tokenizer=tokenizer,
            logger=self.logger,
        )

        self._data_collator = default_data_collator

        if self.model is not None:
            self.pad_token_id, self.eos_token_id = get_pad_eos(
                self.tokenizer, self.model
            )
            self._prep_model()

    def _get_call_thread_streamer(self, text: str, args: GENCallArgs = GENCallArgs()):
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.to(self.model.device)

        attention_mask = torch.ones_like(
            input_ids, dtype=torch.long, device=self.model.device
        )

        gen_streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=False
        )

        gen_kwargs = generate_gen_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            streamer=gen_streamer,
            args=args,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

        gen_thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        return gen_thread, gen_streamer

    def __call__(
        self, text: str, args: GENCallArgs = GENCallArgs()
    ) -> GENResult:
        self._get_model_ready_inference()
        gen_thread, gen_streamer = self._get_call_thread_streamer(text, args)
        gen_thread.start()
        out_text = []
        try:
            for stream_result in gen_streamer:
                if stream_result:
                    out_text.append(stream_result)
        finally:
            gen_thread.join()

        final_text = "".join(out_text)

        return GENResult(text=final_text)

    def _tok_function(
        self,
        raw_dataset,
        args: GENTokArgs = GENTokArgs(),
        file_type: str = "jsonl",
        procs: Optional[int] = None,
    ) -> Dataset:
        max_in_len = get_max_in_len(args.max_len, self.tokenizer)

        return tokenize_gen(
            tokenizer=self.tokenizer,
            dataset=raw_dataset,
            max_len=max_in_len,
            bs=args.bs,
            procs=procs,
        )

    def train(
        self,
        train_path: str = "",
        args: EricTrainArgs = EricTrainArgs(),
        eval_path: str = "",
        resume_path: str = "",
    ):
        return super(EricGeneration, self).train(
            train_path, args, eval_path, resume_path=resume_path
        )

    def eval(
        self, eval_path: str = "", args: EricEvalArgs = EricEvalArgs()
    ) -> EvalResult:
        return super(EricGeneration, self).eval(
            eval_path=eval_path, args=args
        )

    def tok(self, path: str, out_dir: str, args: GENTokArgs = GENTokArgs()):
        return super(EricGeneration, self).tok(
            path=path, out_dir=out_dir, args=args
        )

    def _load_model_components(
        self,
    ) -> Tuple[PretrainedConfig, PreTrainedTokenizerBase, PreTrainedModel]:
        return get_model_components(
            model_name_path=self.eric_args.model_name,
            trust_remote_code=self.eric_args.trust_remote_code,
            model_class=self.eric_args.model_class,
            tokenizer_path=self.eric_args.tokenizer,
            precision=self.precision_type,
        )

    def _format_tokenized_example(self, example: dict) -> dict:
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "labels": example["labels"],
        }

    def _get_default_eval_models(self) -> List[EvalModel]:
        return []

    def _get_model_ready(self):
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is not None:
            pad_id = self.tokenizer.pad_token_id
        elif self.tokenizer.eos_token_id is not None:
            pad_id = self.tokenizer.eos_token_id
        else:
            raise EricInferenceError(
                "Tokenizer doesn't have a pad_token_id or eos_token_id token"
            )

        if self.model.config.eos_token_id is not None:
            eos_id = self.model.config.eos_token_id
        elif self.tokenizer.eos_token_id is not None:
            eos_id = self.tokenizer.eos_token_id
        else:
            raise EricInferenceError(
                "The model and the tokenizer don't't define an eos_token_id"
            )
        return pad_id, eos_id

    def _prep_model(self):
        generation_config = GenerationConfig.from_model_config(self.model.config)
        args = GENCallArgs()
        generation_config.num_beams = 1
        generation_config.early_stopping = False
        generation_config.do_sample = True
        generation_config.min_len = args.min_len
        generation_config.max_len = args.max_len
        generation_config.temp = args.temp
        generation_config.top_p = args.top_p
        self.model.generation_config = generation_config

    def _get_readme(self, repo_id: str) -> str:
        readme_text = textwrap.dedent(f"""\
        ---
        tags:
        - erictransformer
        - eric-generation
        ---
        
        # {repo_id}

        ## Installation

        ```
        pip install erictransformer
        ```

        ## Usage 

        ```python
        from erictransformer import EricGeneration, GENCallArgs

        eric_gen = EricGeneration(model_name="{repo_id}")
        
        result = eric_gen('Hello world')

        print(result.text)
        
        # Streaming is also possible (see docs)
        ```

        See Eric Transformer's [GitHub](https://github.com/ericfillion/erictransformer) for more information. 
        """)

        return readme_text
