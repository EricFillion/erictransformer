import argparse
import json

from datasets import load_dataset

from erictransformer import EricTextClassification, EricTrainArgs


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Generic
    p.add_argument("--model_name", type=str, default="bert-base-uncased")

    # Number of cases
    p.add_argument("--train_cases", type=int, default=2048)
    p.add_argument("--eval_cases", type=int, default=256)

    # Learning parameters
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--eval_bs", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--gas", type=int, default=1)
    p.add_argument("--optim", choices=["adamw", "sgd"], type=str, default="adamw")
    p.add_argument(
        "--lr_sched",
        choices=["warmup_then_decay", "constant"],
        type=str,
        default="constant",
    )

    # Action Steps
    p.add_argument("--eval_steps", type=int, default=256)
    p.add_argument("--log_steps", type=int, default=256)
    p.add_argument("--checkpoint_steps", type=int, default=-1)
    p.add_argument("--save_best", action="store_true")

    # Paths
    p.add_argument("--train_path", type=str, default="data/train.jsonl")
    p.add_argument("--eval_path", type=str, default="data/eval.jsonl")
    p.add_argument("--save_model_path", type=str, default="model/")
    p.add_argument("--out_dir", type=str, default="eric_transformer/")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--resume_path", type=str, default="")

    # Misc
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main(args):
    if args.train_cases > 43_410:
        raise ValueError("train_cases should be less than or equal to 43410")
    if args.eval_cases > 5_426:
        raise ValueError("eval_cases should be less than or equal to 5426")

    train_dataset = load_dataset(
        "google-research-datasets/go_emotions", split=f"train[0:{args.train_cases}]"
    )

    eval_dataset = load_dataset(
        "google-research-datasets/go_emotions", split=f"validation[0:{args.eval_cases}]"
    )

    labels = [
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relief",
        "remorse",
        "sadness",
        "surprise",
    ]  # skip neutral

    format_data(args.train_path, train_dataset)
    format_data(args.eval_path, eval_dataset)

    eric_tc = EricTextClassification(
        model_name=args.model_name, labels=labels
    )

    train_args = EricTrainArgs(
        out_dir=args.out_dir,
        bs=args.bs,
        eval_bs=args.eval_bs,
        lr=args.lr,
        epochs=args.epochs,
        gas=args.gas,
        optim=args.optim,
        lr_sched=args.lr_sched,
        eval_steps=args.eval_steps,
        log_steps=args.log_steps,
        checkpoint_steps=args.checkpoint_steps,
        save_best=args.save_best,
        seed=args.seed,
        run_name=args.run_name,
    )

    eric_tc.train(
        args.train_path,
        resume_path=args.resume_path,
        args=train_args,
        eval_path=args.eval_path,
    )
    eric_tc.save(args.save_model_path)
    print("model saved to {}".format(args.save_model_path))


def format_data(jsonl_path, dataset):
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for case in dataset:
            text = case["text"]
            for label in case["labels"]:
                if label != 27:  # skip neutral
                    row = {"text": text, "label": label}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(get_args())
