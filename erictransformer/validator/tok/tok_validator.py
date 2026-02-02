from erictransformer.args import TokArgs
from erictransformer.validator import EricValidator


class TokValidator(EricValidator):
    def __init__(self, input_data: str, out_dir: str, args: TokArgs):
        self.input_data = input_data
        self.out_dir = out_dir
        self.args = args
        super().__init__()

    def validate_init(self):
        self._validate_input_data()

    def _validate_input_data(self):
        pass

    def _validate_out_dir(self):
        pass

    def _validate_args(self):
        if not isinstance(self.args.procs, int) and self.args.procs < 0:
            raise ValueError("procs must be an integer greater than or equal to 0")
