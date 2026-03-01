import copy
import os
from logging import Logger
from typing import Union

from transformers import PreTrainedModel

from erictransformer.args import EricTrainArgs
from erictransformer.exceptions import EricInputError
from erictransformer.validator import EricValidator


class TrainValidator(EricValidator):
    def __init__(
        self,
        logger: Logger,
        train_path: str = "",
        args: EricTrainArgs = EricTrainArgs(),
        eval_path: str = "",
        resume_path: str = "",
        model: Union[PreTrainedModel, None] = None,
    ):
        self.train_path = train_path
        self.eval_path = eval_path
        self.resume_path = resume_path
        self.args = copy.deepcopy(args)
        self.logger = logger
        self.model = model

        super().__init__()

        self.train_source = self.resolve_train_source()
        self.eval_source = self.resolve_eval_source()

    def validate_init(self):
        self._validate_required_inputs()
        self._validate_model_or_resume_path()
        self._validate_file_paths_exist()
        self._validate_args_type()
        self._validate_train_args()

    def _validate_required_inputs(self):
        if not self.train_path and not self.resume_path:
            raise EricInputError(
                "You must provide at least one of: `train_path` or `resume_path`."
            )

    def _validate_model_or_resume_path(self):
        if self.model is None and not self.resume_path:
            raise ValueError(
                "Either load a model when calling init or provide resume_path"
            )

    def _validate_file_paths_exist(self):
        if self.train_path and not (
            os.path.isfile(self.train_path) or os.path.isdir(self.train_path)
        ):
            raise EricInputError(
                f"Input file or tok dir not found: {self.train_path}"
            )
        if self.eval_path and not (
            os.path.isfile(self.eval_path) or os.path.isdir(self.eval_path)
        ):
            raise EricInputError(f"Eval file not found: {self.eval_path}")
        if self.resume_path and not os.path.isdir(self.resume_path):
            raise EricInputError(
                f"Resume path directory not found: {self.resume_path}"
            )

    def _validate_args_type(self):
        if not isinstance(self.args, EricTrainArgs):
            raise TypeError(
                f"Expected args to be of type TrainArgs but got {type(self.args).__name__}"
            )


    def _validate_train_args(self):
        args = self.args

        if args.lr <= 0 or args.lr > 1e-2:
            raise EricInputError("`lr` must be > 0 and <= 1e-2.")

        if args.epochs <= 0 or not isinstance(args.epochs, int):
            raise EricInputError("`epochs` must be and int >= 1.")

        if args.gas <= 0 or not isinstance(
            args.gas, int
        ):
            raise EricInputError("`gas` must be and int  > 1.")

        if args.bs <= 0 or not isinstance(args.bs, int):
            raise EricInputError("`bs` must be an int > 0.")

        if args.eval_bs <= -1 or not isinstance(args.eval_bs, int):
            raise EricInputError("`eval_bs` must be and int >= 0.")

        if args.eval_steps < 1 and args.eval_steps != 0:
            self.logger.warning(
                f"eval_steps must be an integer greater than 1, but got {args.eval_steps}. eval_steps will be set to 256."
            )
            args.eval_steps = 256

        if args.log_steps < 1 and args.log_steps != 0:
            self.logger.warning(
                f"log_steps must be an integer greater than 1, but got {args.log_steps}. log_steps will be set to 256."
            )
            args.log_steps = 256

        if args.checkpoint_steps < 1:
            if args.checkpoint_steps != -1:
                raise EricInputError(
                    "`checkpoint_steps` must be an int >= 1. Use -1 to disable checkpointing."
                )
        if args.lr_sched not in ["constant", "warmup_then_decay"]:
            raise EricInputError(
                "`lr_sched` must be 'constant' or 'warmup_then_decay'."
            )

        if not isinstance(args.seed, int) or args.seed < 0:
            raise EricInputError("`seed` must be a non-negative int.")
        if hasattr(args, "project_name"):
            if not isinstance(args.project_name, str):
                raise EricInputError("`project_name` must be a string.")

        if not isinstance(args.run_name, str):
            raise EricInputError("`run_name` must be a string.")

        if not isinstance(args.save_best, bool):
            raise EricInputError("`save_best` must be a boolean.")

    def resolve_train_source(self) -> str:
        if os.path.isfile(self.train_path):
            return "file"
        elif os.path.isdir(self.train_path):
            return "folder"

    def resolve_eval_source(self) -> str:
        if os.path.isfile(self.eval_path):
            return "file"
        elif os.path.isdir(self.eval_path):
            return "folder"
        else:
            return "no_eval"
