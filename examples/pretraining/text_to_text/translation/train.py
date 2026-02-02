import argparse

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from erictransformer import EricTextToText, EricTrainArgs


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Generic
    p.add_argument("--model_name", default="google-t5/t5-small")
    p.add_argument("--tok_train", type=str, default="data/tok/train")
    p.add_argument("--tok_eval", type=str, default="data/tok/eval")

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
        default="warmup_then_decay",
    )

    # Action Steps
    p.add_argument("--eval_steps", type=int, default=256)
    p.add_argument("--log_steps", type=int, default=256)
    p.add_argument("--checkpoint_steps", type=int, default=-1)
    p.add_argument("--save_best", action="store_true")

    # Paths
    p.add_argument("--save_model_path", type=str, default="model/")
    p.add_argument("--out_dir", type=str, default="eric_transformer/")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--resume_path", type=str, default="")

    # Misc
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main(args):
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    eric_tt = EricTextToText(
        model_name=model if not args.resume_path else args.resume_path,
        tokenizer=tokenizer,
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

    eric_tt.train(
        args.tok_train,
        resume_path=args.resume_path,
        eval_path=args.tok_eval,
        args=train_args,
    )

    eric_tt.save(args.save_model_path)
    print("model saved to {}".format(args.save_model_path))


if __name__ == "__main__":
    main(get_args())
