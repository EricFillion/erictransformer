from torch.optim import SGD, AdamW

from erictransformer.args import EricTrainArgs


def get_optim(args: EricTrainArgs, model, logger):
    if args.optim == "adamw":
        logger.debug("Using PyTorch's AdamW optimizer.")
        return AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    elif args.optim == "sgd":
        logger.debug("Using PyTorch's SGD optimizer")
        return SGD(
            model.parameters(), lr=args.lr, momentum=0, weight_decay=1e-4
        )

    else:
        raise ValueError(f"Unsupported optimizer: {args.optim}")
