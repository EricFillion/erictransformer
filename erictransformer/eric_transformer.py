import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from torch.optim.lr_scheduler import ConstantLR
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from erictransformer.args import CallResult, CallArgs, EricEvalArgs, TokArgs, EricTrainArgs
from erictransformer.eval_models import EvalModel
from erictransformer.exceptions import (
    EricDatasetError,
    EricDeviceError,
    EricIOError,
    EricNoModelError,
    EricPushError,
    EricResumeError,
    EricSaveError,
)
from erictransformer.eric_tracker import EricTracker
from erictransformer.loops import EvalResult, TrainResult, eval_loop, train_loop
from erictransformer.utils import (
    EricTimer,
    create_tracker_dir,
    get_procs,
    get_num_training_steps,
    get_optim,
    get_precision,
    get_tok_data,
    et_get_device,
    et_get_logger,
    prepare_output_locations,
    resolve_input_files,
    resume_training,
    save_json_tok_data,
    tok_dir_to_dataset,
    write_details_file,
)
from erictransformer.validator import EvalValidator, TokValidator, TrainValidator


@dataclass
class EricTransformerArgs:
    # Arguments class for EricTransformer.__init__()
    model_name: str
    model_class: AutoModel
    use_auth_token: Union[str, bool, None] = None
    trust_remote_code: bool = False
    tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast, None] = None


class EricTransformer(ABC):
    def __init__(self, eric_args: EricTransformerArgs):
        self.logger = et_get_logger()
        self.eric_args = eric_args
        self.device = et_get_device()
        self.precision_type = get_precision(device=self.device)

        self.config, self.tokenizer, self.model = self._load_model_components()

        if self.model is not None:
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.logger.info("Using device: %s", self.device)

        # These are set in child classes
        self._data_collator = None

        self._train_just_happened = True

        self.eval_models = self._get_default_eval_models()

    @abstractmethod
    def __call__(
        self, text: str, args: CallArgs = CallArgs()
    ) -> List[CallResult]:
        raise NotImplementedError()

    @abstractmethod
    def _tok_function(
        self, raw_dataset, args: TokArgs, file_type: str, procs: int = 1
    ) -> Dataset:
        raise NotImplementedError()

    @abstractmethod
    def _load_model_components(
        self,
    ) -> Tuple[PretrainedConfig, PreTrainedTokenizerBase, PreTrainedModel]:
        pass

    @abstractmethod
    def _format_tokenized_example(self, example: dict) -> dict:
        pass

    @abstractmethod
    def _get_default_eval_models(self) -> List[EvalModel]:
        pass

    @abstractmethod
    def _get_readme(self, repo_id: str) -> str:
        pass

    @abstractmethod
    def _prep_model(self):
        pass

    def train(
        self,
        train_path: str = "",
        args: EricTrainArgs = EricTrainArgs(),
        eval_path: str = "",
        *,
        resume_path: str = "",  # a path to a dir
    ) -> TrainResult:
        out_dir = create_tracker_dir(args.out_dir, "train", args.run_name)

        timer_dir = os.path.join(out_dir, "time")

        eric_timer = EricTimer(out_dir=timer_dir)

        eric_train_validator = TrainValidator(
            self.logger,
            train_path=train_path,
            args=args,
            eval_path=eval_path,
            resume_path=resume_path,
            model=self.model,
        )
        # the validator may alter some of the parameters
        args = eric_train_validator.args

        tracker_state = None
        if resume_path:
            with eric_timer.section("resume", "load resume path"):
                (
                    tracker_state,
                    args_dict,
                    model_tokenizer_path,
                    lr_sched_path,
                ) = resume_training(resume_path)
                args = EricTrainArgs(**args_dict)

                self.config, self.tokenizer, self.model = self._load_model_components()

        if eric_train_validator.train_source == "file":
            with eric_timer.section("tokenize", "tokenize_train"):
                current_train_tok_dir = self._tokenize_input_file(
                    train_path, out_dir, "train"
                )
            self.logger.info(
                f"Train data has been tokenized and saved to {current_train_tok_dir}"
            )

        else:  # has to be "train_tok_dir"
            current_train_tok_dir = train_path

        with eric_timer.section("tokenize", "load_train_data"):
            train_dataloader, num_train_cases = get_tok_data(
                current_train_tok_dir,
                args.seed,
                args.bs,
                self._data_collator,
                self.device,
            )

        skip_eval = False
        if eric_train_validator.eval_source == "file":
            with eric_timer.section("tokenize", "tokenize_eval"):
                current_eval_tok_dir = self._tokenize_input_file(
                    eval_path, out_dir, "eval"
                )
            self.logger.info(
                f"Eval data has been tokenized and saved to {current_eval_tok_dir}"
            )
        elif eric_train_validator.eval_source == "folder":
            current_eval_tok_dir = eval_path
        else:
            self.logger.info(
                "No evaluating data will be used. Provide eval_tok_dir or eval_filepath"
            )
            skip_eval = True

        if not skip_eval:
            with eric_timer.section("tokenize", "load_eval_data"):
                eval_dataloader, num_eval_cases = get_tok_data(
                    current_eval_tok_dir,
                    args.seed,
                    args.eval_bs if args.eval_bs else args.bs,
                    self._data_collator,
                    self.device,
                )
        else:
            num_eval_cases = 0
            eval_dataloader = None

        try:
            self.model.to(self.device)
        except Exception as e:
            raise EricDeviceError(f"Failed to move model to {e}")

        train_steps = get_num_training_steps(
            train_cases=num_train_cases,
            num_devices=1,
            epochs=args.epochs,
            gas=args.gas,
            bs=args.bs,
        )

        eric_tracker = EricTracker(
            args, train_steps, out_dir, tracker_state if resume_path else None
        )

        optim = get_optim(args, self.model, self.logger)

        lr_sched = None
        if args.lr_sched == "warmup_then_decay":
            if train_steps < 8:
                self.logger.info(
                    "You need to have at least 8 steps to use the 'warmup_then_decay' lr_sched. Falling back to 'constant'"
                )
            else:
                num_warmup_steps = math.ceil(train_steps / 10)  # 10% warmup

                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optim,
                    start_factor=0.1,  # 10% of the steps are for the warmup phase
                    end_factor=1.0,
                    total_iters=num_warmup_steps,
                )

                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim,
                    T_max=train_steps - num_warmup_steps,
                    eta_min=0.1 * args.lr,
                )

                lr_sched = torch.optim.lr_scheduler.SequentialLR(
                    optim,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[num_warmup_steps],
                )

        if lr_sched is None:
            lr_sched = ConstantLR(optim, factor=1, total_iters=1)

        skip_cases = 0
        starting_epoch = 1

        if resume_path:
            cases_per_step = args.bs * args.gas
            if tracker_state["last_checkpoint_step"] == train_steps:
                raise EricResumeError(
                    "You provided a path to an already completed training run"
                )
            skip_cases = tracker_state["last_checkpoint_step"] * cases_per_step
            current_epoch = tracker_state["epoch"]
            starting_epoch = current_epoch
            if current_epoch > 1:
                skip_cases = skip_cases - (current_epoch - 1) * num_train_cases
            try:
                with eric_timer.section("resume", "load lr scheduler"):
                    lr_sched.load_state_dict(torch.load(lr_sched_path))
                    # for now we don't resume the optimizer's state as it was causing problems.
            except Exception as e:
                raise EricResumeError(
                    f"Could not load lr scheduler state from resume path: {e}"
                )

        self.logger.info(f"Model on {next(self.model.parameters()).device} ")

        self.model.train()

        train_loop_result: TrainResult = train_loop(
            args=args,
            train_dataloader=train_dataloader,
            eric_tracker=eric_tracker,
            train_steps=train_steps,
            model=self.model,
            optim=optim,
            lr_sched=lr_sched,
            eval_cases=num_eval_cases,
            eval_dataloader=eval_dataloader,
            tokenizer=self.tokenizer,
            config=self.config,
            skip_cases=skip_cases,
            starting_epoch=starting_epoch,
            eval_models=self.eval_models,
            eric_timer=eric_timer,
            precision_type=self.precision_type,
        )

        eric_tracker.close()

        self._train_just_happened = True
        eric_timer.report()
        return train_loop_result

    def eval(self, eval_path: str = "", args: EricEvalArgs = EricEvalArgs()) -> EvalResult:
        eval_validator = EvalValidator(
            model=self.model,
            eval_path=eval_path,
            args=args,
            out_dir=args.out_dir,
            logger=self.logger,
        )

        if eval_validator.eval_source == "file":
            out_directory = create_tracker_dir(
                args.out_dir, "eval", args.run_name
            )
            current_eval_tok_dir = self._tokenize_input_file(
                eval_path, out_directory, "eval", in_eval=True
            )
            self.logger.info(
                f"Eval data has been tokenized and saved to {current_eval_tok_dir}"
            )
        else:  # has to be train_path:
            current_eval_tok_dir = eval_path

        dataloader, tok_data_len = get_tok_data(
            current_eval_tok_dir,
            args.seed,
            args.bs,
            self._data_collator,
            self.device,
        )

        try:
            self.model.to(self.device)
        except Exception as e:
            raise EricDeviceError(f"Failed to move model to {e}")

        self.model.eval()
        return eval_loop(
            self.model,
            dataloader,
            tok_data_len,
            args.bs,
            eval_models=self.eval_models,
        )

    def tok(self, path: str, out_dir: str, args: TokArgs = TokArgs()):
        _ = TokValidator(
            input_data=path, out_dir=out_dir, args=args
        )

        file_infos = resolve_input_files(path)

        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            raise EricIOError(f"Failed to make directory: {out_dir}. {e}")
        output_paths, output_data_loc = prepare_output_locations(
            out_dir, args.shards
        )

        num_cases = 0
        p_num_files = tqdm(
            total=len(file_infos), desc="# tokenizing shard #", position=0
        )

        for file, file_type in file_infos:
            if args.max_cases >0 and num_cases >= args.max_cases:
                break
            try:
                if file_type == "text":
                    dataset = load_dataset(
                        file_type,
                        data_files=[file],
                        split="train",
                        sample_by="document",
                    )
                else:
                    dataset = load_dataset(file_type, data_files=[file], split="train")

            except Exception as e:
                raise (
                    EricDatasetError(f"Error loading {file} of type {file_type}: {e}")
                )

            # Trim dataset if necessary
            remaining = (
                args.max_cases - num_cases
                if args.max_cases > 0
                else len(dataset)
            )
            trimmed_count = min(len(dataset), remaining)
            dataset = dataset.select(range(trimmed_count))
            procs = get_procs(args.procs)
            # Tokenize trimmed data
            tokenized = self._tok_function(
                dataset, args=args, file_type=file_type, procs=procs
            )

            save_json_tok_data(
                tokenized,
                output_data_loc,
                args.shards,
                self._format_tokenized_example,
            )

            num_cases += len(tokenized)

            p_num_files.update(1)

        write_details_file(out_dir, num_cases, output_paths)

        for f in output_data_loc:
            f.close()

    def save(self, path: str):
        if self.model is None:
            raise EricNoModelError(
                f"No model found. Provide a model_name or PreTrainedModel when instantiating {self.__class__.__name__}."
            )
        try:
            self.model.save_pretrained(path)
        except Exception as e:
            raise EricSaveError(f"Error saving model to {path}: {e}")
        try:
            self.tokenizer.save_pretrained(path)
        except Exception as e:
            raise EricSaveError(f"Error saving tokenizer to {path}: {e}")
        try:
            self.config.save_pretrained(path)
        except Exception as e:
            raise EricSaveError(f"Error saving config to {path}: {e}")

        self._prep_model()  # Reset self.model.generation_config with the Eric Transformer values for CHAT, GEN and TT

    def push(self, repo_id: str, private: bool = True):
        if self.model is None:
            raise EricNoModelError(
                f"No model found. Provide a model_name or PreTrainedModel when instantiating {self.__class__.__name__}."
            )

        api = HfApi()
        try:
            api.create_repo(repo_id, exist_ok=True, private=private)
        except Exception as e:
            self.logger.warning(f"Could not crate repo {e}")
            return
        try:
            has_readme = api.file_exists(repo_id, "README.md")
        except Exception as e:
            self.logger.warning(f"Could not info: {e}")
            return

        if not has_readme:
            readme_text = self._get_readme(repo_id)
            try:
                self.logger.info("Pushing README...")

                api.upload_file(
                    path_or_fileobj=readme_text.encode("utf-8"),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                )

            except Exception as e:
                # Don’t fail the whole push if README upload fails; just warn.
                self.logger.warning(f"Error pushing README: {e}")

        try:
            self.logger.info("Pushing model...")
            self.model.push_to_hub(
                repo_id,
                private=private,
                commit_message="Uploaded model from Eric Transformer",
            )
        except Exception as e:
            raise EricPushError(f"Error pushing model: {e}")
        try:
            self.logger.info("Pushing tokenizer...")
            self.tokenizer.push_to_hub(
                repo_id,
                private=private,
                commit_message="Uploaded tokenizer from Eric Transformer",
            )
        except Exception as e:
            raise EricPushError(f"Error pushing tokenizer: {e}")
        try:
            self.logger.info("Pushing config...")
            self.config.push_to_hub(
                repo_id,
                private=private,
                commit_message="Uploaded config from Eric Transformer",
            )
        except Exception as e:
            raise EricPushError(f"Error pushing config: {e}")

    def _tokenize_input_file(
        self, train_path: str, out_dir: str, label: str, in_eval: bool = False
    ) -> str:
        dir_name = f"tok_{label}_data" if in_eval else f"data/tok_{label}_data"
        tok_dir = os.path.join(out_dir, dir_name)
        try:
            os.makedirs(tok_dir, exist_ok=True)
        except Exception as e:
            raise EricIOError(f"error making directory ({tok_dir}): {e}")

        self.tok(train_path, tok_dir)
        self.logger.info(
            f"{label.capitalize()} data has been tokenized and saved to {tok_dir}"
        )
        return tok_dir

    def _get_model_ready_inference(self):
        if self._train_just_happened:
            self.logger.info(f"Moving model to {self.device}")
            try:
                self.model.to(self.device)
            except Exception as e:
                raise EricDeviceError(f"Failed to move model to device: {e}")

        self._train_just_happened = False

    @staticmethod
    def tok_dir_to_dataset(tok_dir: str) -> Dataset:
        return tok_dir_to_dataset(tok_dir)
