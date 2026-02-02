import argparse

from transformers import AutoConfig, AutoModelForCausalLM

from erictransformer import EricTrainArgs, EricGeneration


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Generic
    p.add_argument("--model_name", type=str, default="cerebras/Cerebras-GPT-111M")

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
    p.add_argument("--tok_train", type=str, default="data/tok/train")
    p.add_argument("--tok_eval", type=str, default="data/tok/eval")
    p.add_argument("--save_model_path", type=str, default="model/")
    p.add_argument("--out_dir", type=str, default="eric_transformer/")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--resume_path", type=str, default="")

    # Misc
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main(args):
    if args.bs == 1:
        print("Consider increasing --bs if you have enough memory.")

    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_config(config)

    eric_gen = EricGeneration(model_name=model, tokenizer=args.model_name)

    train_args = EricTrainArgs(
        out_dir=args.out_dir,
        epochs=args.epochs,
        bs=args.bs,
        eval_bs=args.eval_bs,
        lr=args.lr,
        optim=args.optim,
        lr_sched=args.lr_sched,
        eval_steps=args.eval_steps,
        log_steps=args.log_steps,
        checkpoint_steps=args.checkpoint_steps,
        save_best=args.save_best,
        run_name=args.run_name,
    )

    train_result = eric_gen.train(
        args.tok_train,
        resume_path=args.resume_path,
        eval_path=args.tok_eval,
        args=train_args,
    )

    eric_gen.save(args.save_model_path)

    print("model saved to {}".format(args.save_model_path))


if __name__ == "__main__":
    main(get_args())
