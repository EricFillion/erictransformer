import argparse

from erictransformer import EricTextToText, TTTokArgs


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="google-t5/t5-small")

    parser.add_argument("--tok_train", type=str, default="data/tok/train")

    parser.add_argument("--untok_train", type=str, default="data/untok/train")

    parser.add_argument("--tok_eval", type=str, default="data/tok/eval")

    parser.add_argument("--untok_eval", type=str, default="data/untok/eval")

    parser.add_argument("--shards", type=int, default=32)

    parser.add_argument("--bs", type=int, default=1024)

    parser.add_argument("--procs", type=int, default=0)

    return parser.parse_args()


def main(args):
    happy = EricTextToText(model_name=None, tokenizer=args.model_name)
    tok_args = TTTokArgs(
        shards=args.shards, bs=args.bs, procs=args.procs
    )

    happy.tok(path=args.untok_train, out_dir=args.tok_train, args=tok_args)
    happy.tok(path=args.untok_eval, out_dir=args.tok_eval, args=tok_args)


if __name__ == "__main__":
    main(get_args())
