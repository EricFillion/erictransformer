import glob
import json
import os
from datetime import datetime
from typing import List, Tuple, Union

from datasets import Dataset

from erictransformer.exceptions import EricInputError, EricIOError

EXT_TO_TYPE_MAP = {
    ".json": "json",
    ".jsonl": "json",  # jsonl is loaded as json
}


def resolve_input_files(input_data: Union[str, Dataset]) -> List[tuple]:
    try:
        if isinstance(input_data, str):
            if os.path.isdir(input_data):
                return collect_files_from_dir(input_data)
            elif os.path.isfile(input_data):
                ext = os.path.splitext(input_data)[1].lower()
                if ext not in EXT_TO_TYPE_MAP:
                    raise EricInputError(f"Unsupported file extension: {ext}")
                return [(input_data, EXT_TO_TYPE_MAP[ext])]
            else:
                raise EricInputError(f"Path does not exist: {input_data}")
        else:
            raise EricInputError("Input must be a file path or directory string.")
    except Exception as e:
        raise EricInputError(f"Failed to resolve input files: {e}")


def collect_files_from_dir(dir_path: str) -> List[Tuple[str, str]]:
    try:
        jsonl = [(f, "json") for f in glob.glob(os.path.join(dir_path, "*.jsonl"))]
        # text = [(f, "text") for f in glob.glob(os.path.join(dir_path, "*.txt"))]
        # we might support .txt files for EricGeneration models one day
        all_files = jsonl
        if not all_files:
            raise EricInputError(
                f"No valid jsonl files found in directory: {dir_path}"
            )
        return all_files
    except Exception as e:
        raise EricIOError(f"Failed to collect files from directory '{dir_path}': {e}")


def prepare_output_locations(out_dir: str, shards: int):
    try:
        os.makedirs(out_dir, exist_ok=True)
        paths = [
            os.path.join(out_dir, f"tok_shard_{i + 1}.jsonl")
            for i in range(shards)
        ]
        handles = []
        for p in paths:
            try:
                handles.append(open(p, "w"))
            except OSError as e:
                raise EricIOError(f"Failed to open file for writing: {p} - {e}")
        return paths, handles
    except Exception as e:
        raise EricIOError(f"Failed to prepare output locations in '{out_dir}': {e}")


def write_details_file(out_dir: str, num_cases: int, output_paths: List[str]):
    try:
        detail = {
            "num_cases": num_cases,
            "timestamp": datetime.now().isoformat(),
            "paths": output_paths,
        }
        details_path = os.path.join(out_dir, "eric_details.json")
        with open(details_path, "w") as f:
            json.dump(detail, f, indent=2)
    except Exception as e:
        raise EricIOError(f"Failed to write details file in '{out_dir}': {e}")
