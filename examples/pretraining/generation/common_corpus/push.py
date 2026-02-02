import argparse

from erictransformer import EricGeneration


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", required=True)
    p.add_argument("--save_model_path", default="model/")

    return p.parse_args()


def main(args):
    eric_data = EricGeneration(model_name=args.save_model_path)
    eric_data.push(args.repo_id)


if __name__ == "__main__":
    main(get_args())
