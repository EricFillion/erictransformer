import json
import os

from datasets import Dataset

from erictransformer.exceptions import EricDatasetError, EricInputError, EricIOError


def tok_dir_to_dataset(tok_dir: str):
    # Validate the input directory
    if not os.path.isdir(tok_dir):
        raise EricInputError(f"Invalid tok_dir: {tok_dir} is not a directory.")

    # Attempt to construct the path to eric_details.json
    try:
        eric_details_path = os.path.join(tok_dir, "eric_details.json")
    except Exception as e:
        raise EricInputError(f"Error creating path to eric_details.json: {e}")

    # Read the eric_details.json file
    try:
        with open(eric_details_path, "r") as f:
            eric_details = json.load(f)
    except Exception as e:
        raise EricIOError(f"Unable to read '{eric_details_path}': {e}")

    # List of full paths to all JSONL files
    full_paths = eric_details.get("paths", [])
    if not full_paths:
        raise EricDatasetError("No file paths found in eric_details.json.")

    # Load all entries from each JSONL file
    all_data = []
    for path in full_paths:
        try:
            with open(path, "r") as f:
                for line in f:
                    all_data.append(json.loads(line))
        except Exception as e:
            raise EricIOError(f"Unable to read '{path}': {e}")

    # Convert list of dicts to Hugging Face Dataset
    try:
        dataset = Dataset.from_list(all_data)
    except Exception as e:
        raise EricDatasetError(f"Unable to create dataset: {e}")

    return dataset
