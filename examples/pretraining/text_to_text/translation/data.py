import argparse
import json
import os
import random

from datasets import load_dataset
from tqdm.auto import tqdm


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # UNIQUE
    p.add_argument("--source_lang", default="en")
    p.add_argument("--target_lang", default="fr")
    p.add_argument("--min_char_length", type=int, default=32)

    # GENERIC
    p.add_argument("--train_cases", type=int, default=524_288)
    p.add_argument("--eval_cases", type=int, default=256)
    p.add_argument(
        "--shards", type=int, default=32, help="How many JSONL shards to write"
    )

    p.add_argument("--untok_train", type=str, default="data/untok/train/")
    p.add_argument("--untok_eval", type=str, default="data/untok/eval/")

    p.add_argument("--seed", type=int, default=42)

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


def shard_sizes(total: int, k: int):
    base = total // k
    remainder = total % k
    return [base + (1 if i < remainder else 0) for i in range(k)]


def main(args):
    random.seed(args.seed)

    original_id = f"{args.source_lang}-{args.target_lang}"
    original_id_flipped = f"{args.source_lang}-{args.target_lang}"

    if args.source_lang == "en":
        english_is_source = True
    else:
        english_is_source = False

    if original_id in NUM_CASES:
        get_id = original_id
    elif original_id_flipped in NUM_CASES:
        get_id = original_id_flipped
    else:
        raise ValueError(
            f"Invalid source/target languages. Valid language pairs include: {list(NUM_CASES.keys())}"
        )

    max_cases = NUM_CASES[get_id]

    total_needed = args.train_cases + args.eval_cases
    if total_needed > max_cases:
        raise ValueError(
            f"Too many cases requested ({total_needed}). "
            f"Maximum available for {get_id} is {max_cases}."
        )

    train_quota = shard_sizes(args.train_cases, args.shards)
    eval_quota = shard_sizes(args.eval_cases, args.shards)

    train_files, train_written = make_writers(
        args.untok_train, "train", args.shards
    )
    eval_files, eval_written = make_writers(args.untok_eval, "eval", args.shards)

    t_idx = e_idx = 0
    eval_chance = args.eval_cases / total_needed

    ds = load_dataset(
        "sentence-transformers/parallel-sentences-ccmatrix",
        get_id,
        split="train",
        streaming=True,
    )

    p_train = tqdm(total=args.train_cases, desc="Train written", position=0)
    p_eval = tqdm(total=args.eval_cases, desc="Eval written", position=1)
    p_scan = tqdm(
        total=None, desc="Samples scanned", position=2
    )

    try:
        for row in ds:
            # scanned (every pulled sample)
            p_scan.update(1)

            # stop condition if we have all quotas
            if (
                sum(train_written) == args.train_cases
                and sum(eval_written) == args.eval_cases
            ):
                break
            if not row:
                continue

            if english_is_source:
                src = row.get("english")
                tgt = row.get("non_english")
            else:
                # if flipped, we invert so that output is still (input->output)
                src = row.get("non_english")
                tgt = row.get("english")

            if not src or not tgt:
                continue
            if len(src) < args.min_char_length or len(tgt) < args.min_char_length:
                continue

            if sum(eval_written) < args.eval_cases and random.random() < eval_chance:
                while (
                    e_idx < args.shards and eval_written[e_idx] >= eval_quota[e_idx]
                ):
                    e_idx += 1
                if e_idx >= args.shards:
                    continue
                eval_files[e_idx].write(
                    json.dumps({"input": src, "target": tgt}, ensure_ascii=False) + "\n"
                )
                eval_written[e_idx] += 1
                p_eval.update(1)
            else:
                if sum(train_written) >= args.train_cases:
                    continue
                while (
                    t_idx < args.shards
                    and train_written[t_idx] >= train_quota[t_idx]
                ):
                    t_idx += 1
                if t_idx >= args.shards:
                    continue
                train_files[t_idx].write(
                    json.dumps({"input": src, "target": tgt}, ensure_ascii=False) + "\n"
                )
                train_written[t_idx] += 1
                p_train.update(1)

            p_scan.set_postfix(
                {
                    "train_total": sum(train_written),
                    "eval_total": sum(eval_written),
                    "train_shard": f"{min(t_idx + 1, args.shards)}/{args.shards}",
                    "eval_shard": f"{min(e_idx + 1, args.shards)}/{args.shards}",
                }
            )

    finally:
        close_all(train_files + eval_files)
        p_train.close()
        p_eval.close()
        p_scan.close()

    print(
        f"Wrote {args.train_cases} train cases across {args.shards} shards -> {args.untok_train}"
    )
    print(
        f"Wrote {args.eval_cases} eval  cases across {args.shards} shards -> {args.untok_eval}"
    )


NUM_CASES = {
    "en-af": 8694461,
    "en-ar": 49697322,
    "en-ast": 2956618,
    "en-az": 1251254,
    "en-be": 1885446,
    "en-bg": 44635282,
    "en-bn": 10074620,
    "en-br": 454175,
    "en-ca": 21284430,
    "en-ceb": 962549,
    "en-cs": 56307029,
    "en-da": 52273664,
    "en-de": 247470736,
    "en-el": 49262631,
    "en-eo": 15418393,
    "en-es": 409061333,
    "en-et": 22007049,
    "en-eu": 7778871,
    "en-fa": 24597533,
    "en-fi": 35982562,
    "en-fr": 328595738,
    "en-fy": 1372321,
    "en-ga": 1076420,
    "en-gd": 310351,
    "en-gl": 13178507,
    "en-ha": 5861080,
    "en-he": 25228938,
    "en-hi": 15127900,
    "en-hr": 18797643,
    "en-hu": 36435409,
    "en-id": 70545705,
    "en-ig": 80385,
    "en-ilo": 335469,
    "en-is": 8723145,
    "en-it": 146240552,
    "en-ja": 40883733,
    "en-jv": 819280,
    "en-ko": 19358582,
    "en-la": 1114190,
    "en-lb": 11978495,
    "en-lt": 23298470,
    "en-lv": 16685969,
    "en-mg": 1736359,
    "en-mk": 12040173,
    "en-ml": 6809956,
    "en-mr": 2874211,
    "en-ms": 10730648,
    "en-ne": 708316,
    "en-nl": 106695917,
    "en-no": 47801406,
    "en-oc": 1730828,
    "en-or": 96595,
    "en-pl": 74070714,
    "en-pt": 173743166,
    "en-ro": 55607023,
    "en-ru": 139937785,
    "en-sd": 1717573,
    "en-si": 6270800,
    "en-sk": 38096241,
    "en-sl": 27406782,
    "en-so": 222793,
    "en-sq": 22358158,
    "en-sr": 26510872,
    "en-su": 271736,
    "en-sv": 77008059,
    "en-sw": 5756664,
    "en-ta": 7291118,
    "en-tl": 3113828,
    "en-tr": 47045956,
    "en-uk": 20240171,
    "en-ur": 6094149,
    "en-vi": 50092444,
    "en-xh": 18980689,
    "en-yi": 275076,
    "en-zh": 71383325,
}

if __name__ == "__main__":
    main(get_args())
