import torch


def get_precision(device) -> str:
    is_cuda = device.type == "cuda"
    bf16_ok = (
        bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if is_cuda
        else False
    )

    use_bf16 = is_cuda and bf16_ok
    use_fp16 = is_cuda and not use_bf16

    if use_bf16:
        precision_type = "bf16"
    elif use_fp16:
        precision_type = "fp16"
    else:
        precision_type = "fp32"

    return precision_type
