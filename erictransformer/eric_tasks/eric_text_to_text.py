import textwrap
import threading
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Union

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    TextIteratorStreamer,
)

from erictransformer.args import EricTrainArgs, EricEvalArgs
from erictransformer.eval_models import EvalModel
from erictransformer.exceptions import EricTokenizationError
from erictransformer.eric_tasks.args import (
    TTCallArgs,
    TTTokArgs,
)
from erictransformer.eric_tasks.misc import generate_tt_kwargs, get_pad_eos
from erictransformer.eric_tasks.results import TTResult
from erictransformer.eric_tasks.tok.tok_functions import get_max_in_len
from erictransformer.eric_transformer import EricTransformer, EricTransformerArgs
from erictransformer.loops import EvalResult
from erictransformer.utils import get_model_components
from erictransformer.validator import TTValidator


@dataclass(kw_only=True)
class TTStreamResult:
    text: str


class EricTextToText(EricTransformer):
    def __init__(
        self,
        model_name: Union[str, PreTrainedModel, None] = "google-t5/t5-base",
        *,
        trust_remote_code: bool = False,
        tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    ):
        model_class = AutoModelForSeq2SeqLM
        eric_args = EricTransformerArgs(
            model_name=model_name,
            model_class=model_class,
            trust_remote_code=trust_remote_code,
            tokenizer=tokenizer
        )
        super().__init__(eric_args)

        self.task_validator = TTValidator(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            tokenizer=tokenizer,
            logger=self.logger,
        )

        if self.model is not None:
            self.pad_token_id, self.eos_token_id = get_pad_eos(
                self.tokenizer, self.model
            )

            self._prep_model()
            self._data_collator = DataCollatorForSeq2Seq(
                self.tokenizer, model=self.model
            )

    def _get_call_thread_streamer(self, text: str, args: TTCallArgs = TTCallArgs()):
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True).to(
            self.device
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        gen_streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = generate_tt_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            streamer=gen_streamer,
            args=args,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id
        )

        gen_thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)

        return gen_thread, gen_streamer

    def __call__(
        self,
        text: str,
        args: TTCallArgs = TTCallArgs(),
    ) -> TTResult:
        self._get_model_ready_inference()
        self.task_validator.validate_call(text, args)
        gen_thread, gen_streamer = self._get_call_thread_streamer(text, args)
        gen_thread.start()
        out_text = []
        try:
            for text in gen_streamer:
                out_text.append(text)
        finally:
            gen_thread.join()
            pass

        final_text = "".join(out_text)
        return TTResult(text=final_text)

    def stream(
        self, text: str, args: TTCallArgs = TTCallArgs()
    ) -> Iterator[TTStreamResult]:
        self._get_model_ready_inference()
        self.task_validator.validate_call(text, args)

        gen_thread, gen_streamer = self._get_call_thread_streamer(text, args)
        gen_thread.start()
        try:
            for text in gen_streamer:
                yield TTStreamResult(text=text)
        finally:
            gen_thread.join()

    def _tok_function(
        self,
        raw_dataset,
        args: TTTokArgs = TTTokArgs(),
        file_type: str = "",
        procs: Optional[int] = None,
    ) -> Dataset:
        final_max_in_len = get_max_in_len(
            args.max_in_len, self.tokenizer
        )
        final_max_out_len = get_max_in_len(
            args.max_out_len, self.tokenizer
        )

        def __preprocess_function(examples):
            try:
                model_inputs = self.tokenizer(
                    examples["input"],
                    max_length=final_max_in_len,
                    truncation=True,
                    padding="max_length",
                )

                labels = self.tokenizer(
                    examples["target"],
                    max_length=final_max_out_len,
                    truncation=True,
                    padding="max_length",
                )

                model_inputs["labels"] = [
                    [
                        (tok if tok != self.tokenizer.pad_token_id else -100)
                        for tok in seq
                    ]
                    for seq in labels["input_ids"]
                ]
                return model_inputs
            except Exception as e:
                raise EricTokenizationError(
                    f"Tokenization failed during preprocessing: {e}"
                )

        try:
            tok_dataset = raw_dataset.map(
                __preprocess_function,
                batched=True,
                remove_columns=["input", "target"],
                batch_size=args.bs,
                desc="Tokenizing...",
                num_proc=procs,
            )
            tok_dataset.set_format(
                type="torch", columns=["input_ids", "attention_mask", "labels"]
            )
            return tok_dataset
        except Exception as e:
            raise EricTokenizationError(
                f"Failed to apply preprocessing function over dataset: {e}"
            )

    def train(
        self,
        train_path: str = "",
        args: EricTrainArgs = EricTrainArgs(),
        eval_path: str = "",
        resume_path: str = "",
    ):
        return super().train(train_path, args, eval_path, resume_path=resume_path)

    def eval(self, eval_path: str = "", args=EricEvalArgs()) -> EvalResult:
        return super().eval(eval_path=eval_path, args=args)

    def tok(
        self,
        path: str,
        out_dir: str,
        args: TTTokArgs = TTTokArgs()
    ):
        return super().tok(path=path, out_dir=out_dir, args=args)

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

    def _prep_model(self):
        generation_config = GenerationConfig.from_model_config(self.model.config)
        args = TTCallArgs()
        generation_config.num_beams = 1
        generation_config.early_stopping = False
        generation_config.do_sample = True
        generation_config.min_len = args.min_len
        generation_config.max_length = args.max_len
        generation_config.temp = args.temp
        generation_config.top_p = args.top_p
        self.model.generation_config = generation_config

    def _get_readme(self, repo_id: str) -> str:
        readme_text = textwrap.dedent(f"""\
        ---
        tags:
        - erictransformer
        - eric-text-to-text
        ---
        # {repo_id}

        ## Installation

        ```
        pip install erictransformer
        ```

        ## Usage 

        ```python
        from erictransformer import EricTextToText, TTCallArgs

        eric_tt = EricTextToText(model_name="{repo_id}")

        text = 'Hello world'
        
        result = eric_tt(text)
        print(result.text)
        
        # Stream
        for chunk in eric_tt.stream(text):
            print(chunk.text, end="")
        ```
        
        See Eric Transformer's [GitHub](https://github.com/ericfillion/erictransformer) for more information. 
        """)

        return readme_text
