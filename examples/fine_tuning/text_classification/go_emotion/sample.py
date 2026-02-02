import argparse
import os

from erictransformer import EricTextClassification


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--text", type=str, default="Eric Transformer is a great python package for NLP. It is amazing!"
    )

    p.add_argument("--model_name", type=str, default="model/")

    p.add_argument("--checkpoint", action="store_true")

    return p.parse_args()


def get_checkpoint_path():
    parent_folder = "eric_transformer"
    subfolders = [
        f
        for f in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, f))
    ]

    if not subfolders:
        raise FileNotFoundError("No subfolders found in the specified directory.")

    last_folder = max(subfolders)
    last_folder_path = os.path.join(parent_folder, last_folder)

    checkpoint_path = os.path.join(last_folder_path, "checkpoint")

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"'checkpoint' folder not found in: {last_folder_path}")
    print(f"Will use the model auto detected in {checkpoint_path} ")
    return checkpoint_path


def main(args):
    if args.checkpoint:
        model_name = get_checkpoint_path()
    elif args.model_name:
        model_name = args.model_name
    else:
        raise ValueError(
            "Provide a path to the --model_name parameter or --checkpoint after running train.py with checkpointing enabled."
        )

    print(f"Model: {model_name}")

    eric_tc = EricTextClassification(model_name=model_name)
    result = eric_tc(args.text)

    print("INPUT: ", args.text)
    print("OUTPUT: ", result)  # should be admiration.


if __name__ == "__main__":
    main(get_args())
