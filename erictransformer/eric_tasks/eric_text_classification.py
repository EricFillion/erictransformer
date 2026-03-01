import textwrap
from typing import List, Optional, Tuple, Union

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextClassificationPipeline,
)

from erictransformer.args import EricTrainArgs, EricEvalArgs
from erictransformer.eval_models import EvalModel, TCAccuracyEvalModel
from erictransformer.exceptions import EricInferenceError, EricTokenizationError
from erictransformer.eric_tasks.args import (
    TCCallArgs,
    TCTokArgs,
)
from erictransformer.eric_tasks.inference_engine.text_classification import (
    tc_inference,
)
from erictransformer.eric_tasks.results import TCResult
from erictransformer.eric_tasks.tok.tok_functions import get_max_in_len
from erictransformer.eric_transformer import EricTransformer, EricTransformerArgs
from erictransformer.loops import EvalResult
from erictransformer.utils.init import get_model_components_tc
from erictransformer.validator import TCValidator


class EricTextClassification(EricTransformer):
    def __init__(
        self,
        model_name: Union[str, PreTrainedModel, None] = "bert-base-uncased",
        *,
        trust_remote_code: bool = False,
        tokenizer: Union[str, AutoTokenizer] = None,
        labels: Optional[List[str]] = None
    ):
        model_class = AutoModelForSequenceClassification

        self.labels = labels

        eric_args = EricTransformerArgs(
            model_name=model_name,
            model_class=model_class,
            trust_remote_code=trust_remote_code,
            tokenizer=tokenizer
        )

        super().__init__(eric_args)

        self._pipeline_class = TextClassificationPipeline

        self.task_validator = TCValidator(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            tokenizer=tokenizer,
            logger=self.logger,
            labels=self.labels
        )

        self._data_collator = DataCollatorWithPadding(self.tokenizer)

        self.id2label = self.config.id2label

    def __call__(self, text: str, args: TCCallArgs = TCCallArgs()) -> TCResult:
        self.task_validator.validate_call(text, args)

        self._get_model_ready()

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        ).to(self.device)

        try:
            results = tc_inference(
                tokens=tokens, model=self.model, id2label=self.id2label
            )[0]

        except Exception as e:
            raise EricInferenceError(
                f"Failed to call EricTextClassification's pipeline: {e}"
            )

        labels = []
        scores = []
        for label_and_score in results:
            labels.append(label_and_score[0])
            scores.append(label_and_score[1])
        return TCResult(labels=labels, scores=scores)

    def _tok_function(
        self,
        raw_dataset,
        args: TCTokArgs = TCTokArgs(),
        file_type: str = "",
        procs: Optional[int] = None,
    ) -> Dataset:
        max_in_len = get_max_in_len(args.max_len, self.tokenizer)

        def __preprocess_function(case):
            try:
                result = self.tokenizer(
                    case["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=max_in_len,
                )
                result["labels"] = case["label"]
                return result
            except Exception as e:
                raise EricTokenizationError(
                    f"Tokenization failed during preprocessing: {e}"
                )

        try:
            tok_dataset = raw_dataset.map(
                __preprocess_function,
                batched=True,
                remove_columns=["text", "label"],
                desc="Tokenizing...",
                batch_size=args.bs,
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
        return super(EricTextClassification, self).train(
            train_path, args, eval_path, resume_path=resume_path
        )

    def eval(
        self, eval_path: str = "", args: EricEvalArgs = EricEvalArgs()
    ) -> EvalResult:
        return super(EricTextClassification, self).eval(
            eval_path=eval_path, args=args
        )

    def tok(
        self,
        path: str,
        out_dir: str,
        args: TCTokArgs = TCTokArgs(),
        max_cases: Union[None, int] = None,
    ):
        return super(EricTextClassification, self).tok(
            path=path, out_dir=out_dir, args=args
        )

    def _load_model_components(
        self,
    ) -> Tuple[PretrainedConfig, PreTrainedTokenizerBase, PreTrainedModel]:
        return get_model_components_tc(
            model_name_path=self.eric_args.model_name,
            trust_remote_code=self.eric_args.trust_remote_code,
            model_class=self.eric_args.model_class,
            tokenizer_path=self.eric_args.tokenizer,
            labels=self.labels,
            precision=self.precision_type,
        )

    def _format_tokenized_example(self, example: dict) -> dict:
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "labels": int(example["labels"]),
        }

    def _get_default_eval_models(self) -> List[EvalModel]:
        return [TCAccuracyEvalModel()]

    def _get_model_ready(self):
        self.model = self.model.to(self.device)
        self.model.eval()

    def _prep_model(self):
        pass

    def _get_readme(self, repo_id: str) -> str:
        readme_text = textwrap.dedent(f"""\
        ---
        tags:
        - erictransformer
        - eric-text-classification
        ---
        # {repo_id}

        ## Installation

        ```
        pip install erictransformer
        ```

        ## Usage 

        ```python
        from erictransformer import EricTextClassification

        eric_tc = EricTextClassification(model_name="{repo_id}")

        result = eric_tc('Hello world')

        print(result.labels[0])
        print(result.scores[0])
        ```

        See Eric Transformer's [GitHub](https://github.com/ericfillion/erictransformer) for more information. 
        """)

        return readme_text
