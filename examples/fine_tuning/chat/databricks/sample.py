import argparse
import os

from erictransformer import CHATCallArgs, EricChat


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--text", type=str, default="Tell me about Eric Transformer and a potential use case for it")

    # Train Args Start
    p.add_argument("--model_name", type=str, default="model/")

    p.add_argument("--checkpoint", action="store_true")

    p.add_argument(
        "--context",
        type=str,
        default="Eric Transformer is an open-source Python package for AI that supports pretraining, fine-tuning and retrieval augmented generation (RAG). Many of its components are built with pure PyTorch making it lightweight.",
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
            "Provide a path to the --model_name parameter or enable --checkpoint after running train.py with checkpointing."
        )

    sample_input = [
        {"role": "user", "content": f"{args.context}"},
        {"role": "user", "content": f"{args.text}"},
    ]

    print(f"Loading model from {model_name}")
    eric_chat = EricChat(model_name=model_name)
    result = eric_chat(
        sample_input, args=CHATCallArgs(top_k=8, temp=1)
    )

    print("INPUT: ", sample_input)
    print("OUTPUT: ", result.text)


if __name__ == "__main__":
    main(get_args())
