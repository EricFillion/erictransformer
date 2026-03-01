import argparse
import json
import os
import random

from datasets import load_dataset
from tqdm.auto import tqdm


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_cases", type=int, default=131_072)
    p.add_argument("--eval_cases", type=int, default=256)

    p.add_argument("--untok_train", type=str, default="data/untok/train/")
    p.add_argument("--untok_eval", type=str, default="data/untok/eval/")
    p.add_argument("--save_best", action="store_true")

    # no OpenSource
    default_datasets = [
        "OpenScience",
        "OpenWeb",
        "OpenCulture",
        "OpenGovernment",
        "OpenSemantic",
    ]
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=[
            "OpenScience",
            "OpenWeb",
            "OpenCulture",
            "OpenSource",
            "OpenGovernment",
            "OpenSemantic",
        ],
        default=default_datasets,
    )
    p.add_argument(
        "--shards", type=int, default=32, help="How many JSONL shards to write"
    )

    p.add_argument("--seed", type=int, default=67)

    return p.parse_args()


def make_writers(base_dir: str, prefix: str, k: int):
    os.makedirs(base_dir, exist_ok=True)
    files, counters = [], []
    for i in range(1, k + 1):
        path = os.path.join(base_dir, f"{prefix}_shard_{i}.jsonl")
        files.append(open(path, "w", encoding="utf-8"))
        counters.append(0)
    return files, counters


def close_all(handles):
    for h in handles:
        h.close()


dataset_map = {
    "OpenScience": "Open Science",
    "OpenWeb": "Open Web",
    "OpenCulture": "Open Culture",
    "OpenSource": "Open Source",
    "OpenGovernment": "Open Government",
    "OpenSemantic": "Semantic data",
}


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    def shard_sizes(total: int, k: int):
        base = total // k
        remainder = total % k
        return [base + (1 if i < remainder else 0) for i in range(k)]

    dataset_list = [dataset_map[d] for d in args.datasets]
    assert dataset_list, "Provide valid datasets to --datasets"

    train_quota = shard_sizes(args.train_cases, args.shards)
    eval_quota = shard_sizes(args.eval_cases, args.shards)

    train_files, train_written = make_writers(
        args.untok_train, "train", args.shards
    )
    eval_files, eval_written = make_writers(args.untok_eval, "eval", args.shards)

    t_idx = e_idx = 0

    total_needed = args.train_cases + args.eval_cases
    ds = load_dataset("PleIAs/common_corpus", split="train", streaming=True)

    eval_chance = args.eval_cases / total_needed

    p_train = tqdm(total=args.train_cases, desc="Train written", position=0)

    p_eval = tqdm(total=args.eval_cases, desc="Eval written", position=1)
    p_scan = tqdm(total=None, desc="Samples scanned", position=2)

    try:
        for sample in ds:
            p_scan.update(1)

            if (
                sum(train_written) == args.train_cases
                and sum(eval_written) == args.eval_cases
            ):
                break

            if "text" not in sample:
                continue
            if sample["open_type"] not in dataset_list:
                continue

            # Only accept english unless it's part of OpenSource
            if (
                sample["open_type"] != dataset_map["OpenSource"]
                and sample.get("language") != "English"
            ):
                continue

            # decide eval/train
            if sum(eval_written) < args.eval_cases and random.random() < eval_chance:
                while eval_written[e_idx] >= eval_quota[e_idx]:
                    e_idx += 1
                eval_files[e_idx].write(
                    json.dumps({"text": sample["text"]}, ensure_ascii=False) + "\n"
                )
                eval_written[e_idx] += 1
                p_eval.update(1)
            else:
                if sum(train_written) >= args.train_cases:
                    continue
                while train_written[t_idx] >= train_quota[t_idx]:
                    t_idx += 1
                train_files[t_idx].write(
                    json.dumps({"text": sample["text"]}, ensure_ascii=False) + "\n"
                )
                train_written[t_idx] += 1
                p_train.update(1)

            p_scan.set_postfix(
                {
                    "t_total": sum(train_written),
                    "e_total": sum(eval_written),
                    "t_shard": f"{t_idx + 1}/{args.shards}",
                    "e_shard": f"{e_idx + 1}/{args.shards}",
                }
            )

    finally:
        close_all(train_files + eval_files)
        p_train.close()
        p_eval.close()
        p_scan.close()

    print(f"Wrote {args.train_cases} train cases across {args.shards} shards")
    print(f"Wrote {args.eval_cases} eval cases across {args.shards} shards")


if __name__ == "__main__":
    main(get_args())
