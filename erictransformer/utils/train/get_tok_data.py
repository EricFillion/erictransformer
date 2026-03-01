import json
from pathlib import Path
from typing import List

from datasets import load_dataset
from torch import device
from torch.utils.data import DataLoader
from transformers import DataCollator

from erictransformer.exceptions import EricDatasetError, EricInputError


def get_tok_data(
    tok_dir: str,
    seed: int,
    bs: int,
    collate_fn: DataCollator,
    device_type: device,
):
    details = _load_details_file(tok_dir)
    paths = _resolve_dataset_paths(details, tok_dir)
    dataset = _load_streaming_dataset(paths, seed)
    dataloader = _build_dataloader(dataset, bs, collate_fn, device_type)

    return dataloader, details["num_cases"]


def _load_details_file(tok_dir: str) -> dict:
    details_filepath = Path(tok_dir) / "eric_details.json"
    if not details_filepath.is_file():
        raise FileNotFoundError(f"Missing required file: {details_filepath}")
    try:
        with open(details_filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise EricInputError(f"Failed to parse eric_details.json: {e}")


def _resolve_dataset_paths(details: dict, tok_dir: str) -> List[Path]:
    try:
        paths = [Path(p) for p in details["paths"]]
    except KeyError:
        raise EricInputError(
            f"'paths' field missing in eric_details.json in {tok_dir}"
        )

    if not all(p.is_file() and p.suffix == ".jsonl" for p in paths):
        raise EricInputError("All dataset paths must exist and be `.jsonl` files")
    return paths


def _load_streaming_dataset(paths: List[Path], seed: int):
    data_files = {"train": [str(p) for p in paths]}
    try:
        ds = load_dataset(
            "json",
            data_files=data_files,
            split="train",
            streaming=True,
        )

        # Probe for at least one example (to handle the "all shards empty" case)
        it = iter(ds)
        try:
            _ = next(it)
        except StopIteration:
            raise EricInputError("No non-empty datasets to stream")

        # Re-create the iterator since we advanced one element when probing
        ds = load_dataset("json", data_files=data_files, split="train", streaming=True)

        ds = ds.shuffle(buffer_size=10_000, seed=seed)
        return ds
    except EricInputError:
        raise
    except Exception as e:
        raise EricDatasetError(
            f"Failed to load streaming dataset from paths {paths}: {e}"
        )


def _build_dataloader(
    dataset, bs: int, collate_fn: DataCollator, device_type: device
):
    try:
        shards = getattr(dataset, "n_shards", 1)
        base_workers = min(4, shards)

        if device_type.type == "mps":
            workers = 0
        else:
            workers = base_workers

        return DataLoader(
            dataset,
            batch_size=bs,
            collate_fn=collate_fn,
            num_workers=workers,
            pin_memory=(device_type.type == "cuda"),
            persistent_workers=(workers > 0),
            prefetch_factor=4 if workers > 0 else None,
        )

    except Exception as e:
        raise EricInputError(f"Failed to create DataLoader: {e}")
