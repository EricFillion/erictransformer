import argparse

from erictransformer import GENTokArgs, EricGeneration


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--model_name", type=str, default="cerebras/Cerebras-GPT-111M")

    p.add_argument("--untok_train", type=str, default="data/untok/train")

    p.add_argument("--untok_eval", type=str, default="data/untok/eval")

    p.add_argument("--tok_train", type=str, default="data/tok/train")

    p.add_argument("--tok_eval", type=str, default="data/tok/eval")

    p.add_argument("--shards", type=int, default=32)
    p.add_argument("--bs", type=int, default=1024)
    p.add_argument("--procs", type=int, default=0)

    return p.parse_args()


def main(args):
    eric_gen = EricGeneration(
        model_name=None, tokenizer=args.model_name
    )  # no model is loaded
    tok_args = GENTokArgs(
        shards=args.shards, bs=args.bs, procs=args.procs
    )

    eric_gen.tok(path=args.untok_train, out_dir=args.tok_train, args=tok_args)
    eric_gen.tok(path=args.untok_eval, out_dir=args.tok_eval, args=tok_args)


if __name__ == "__main__":
    main(get_args())
