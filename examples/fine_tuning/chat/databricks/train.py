import argparse
import json

from datasets import load_dataset

from erictransformer import EricTrainArgs, EricChat


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Generic
    p.add_argument("--model_name", type=str, default="openai/gpt-oss-20b")

    # Number of cases
    p.add_argument("--train_cases", type=int, default=2048)
    p.add_argument("--eval_cases", type=int, default=128)
    p.add_argument("--max_chars", type=int, default=10_000)

    # Learning parameters
    p.add_argument("--bs", type=int, default=1)
    p.add_argument(
        "--eval_bs", type=int, default=0
    )  # when 0 uses the same value as --bs
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--gas", type=int, default=1)
    p.add_argument("--optim", choices=["adamw", "sgd"], type=str, default="sgd")
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


def main(args: argparse.Namespace):
    if args.train_cases + args.eval_cases > 15_011:
        raise ValueError("num_train and num_eval must be no more than 15_011")

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    all_data = ds.train_test_split(test_size=args.eval_cases, seed=42)
    train_cases = all_data["train"].select(range(args.train_cases))

    eval_cases = all_data["test"]

    format_data(args.train_path, train_cases, args)
    format_data(args.eval_path, eval_cases, args)

    eric_chat = EricChat(model_name=args.model_name)

    train_args = EricTrainArgs(
        out_dir=args.out_dir,
        bs=args.bs,
        eval_bs=args.eval_bs,
        lr=args.lr,
        epochs=args.epochs,
        gas=args.gas,
        optim=args.optim,
        eval_steps=args.eval_steps,
        log_steps=args.log_steps,
        checkpoint_steps=args.checkpoint_steps,
        save_best=args.save_best,
        seed=args.seed,
        run_name=args.run_name,
    )
    eric_chat.train(
        args.train_path,
        resume_path=args.resume_path,
        args=train_args,
        eval_path=args.eval_path,
    )
    eric_chat.save(args.save_model_path)
    print("model saved to {}".format(args.save_model_path))


def format_data(jsonl_path, dataset, args):
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for case in dataset:
            if case["context"]:
                row = {
                    "messages": [
                        {"role": "user", "content": case["context"][: args.max_chars]},
                        {
                            "role": "user",
                            "content": case["instruction"][: args.max_chars],
                        },
                        {
                            "role": "assistant",
                            "content": case["response"][: args.max_chars],
                        },
                    ]
                }
            else:
                row = {
                    "messages": [
                        {
                            "role": "user",
                            "content": case["instruction"][: args.max_chars],
                        },
                        {
                            "role": "assistant",
                            "content": case["response"][: args.max_chars],
                        },
                    ]
                }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(get_args())
