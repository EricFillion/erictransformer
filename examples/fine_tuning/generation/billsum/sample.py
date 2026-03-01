import argparse
import os

from erictransformer import EricGeneration


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--model_name", type=str, default="model")

    p.add_argument("--checkpoint", action="store_true")

    p.add_argument(
        "--text", type=str, default="The Open-Source Artificial Intelligence Act"
    )
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
            "Provide a path to the --model_path parameter or --checkpoint after running train.py with checkpointing enabled."
        )

    print(f"Loading model from {model_name}")

    prompt = f"SECTION 1. SHORT TITLE. \n This Act may be cited as ``{args.text}''."
    eric_gen = EricGeneration(model_name=model_name)
    result = eric_gen(
        prompt
    )

    print("INPUT: ", prompt)
    print("OUTPUT: ", result.text)


if __name__ == "__main__":
    main(get_args())
