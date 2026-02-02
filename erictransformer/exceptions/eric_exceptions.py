class EricTransformerError(Exception):
    pass


class EricInputError(EricTransformerError):
    pass


class EricLoadModelError(EricTransformerError):
    pass


class EricLoadTokenizerError(EricTransformerError):
    pass


class EricLoadPipelineError(EricTransformerError):
    pass


class EricResumeError(EricTransformerError):
    pass


class EricInferenceError(EricTransformerError):
    pass


class EricDeviceError(EricTransformerError):
    pass


class EricIOError(EricTransformerError):
    pass


class EricTokenizationError(EricTransformerError):
    pass


class EricDatasetError(EricTransformerError):
    pass


class EricChatTemplateError(EricTransformerError):
    pass


class EricEvalModelError(EricTransformerError):
    pass


class EricSaveError(EricTransformerError):
    pass


class EricNoModelError(EricTransformerError):
    pass


class EricPushError(EricTransformerError):
    pass


class EricPlotError(EricTransformerError):
    pass


class EricEvalError(EricTransformerError):
    pass


class EricTrainError(EricTransformerError):
    pass
