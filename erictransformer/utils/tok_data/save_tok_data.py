import json

from erictransformer.exceptions import EricDatasetError, EricIOError


def save_json_tok_data(tok_dataset, output_files, shards, formatter):
    total = len(tok_dataset)
    per_shard = total // shards
    remainder = total % shards

    index = 0

    for shard_id, output_file in enumerate(output_files):
        count = per_shard + (1 if shard_id < remainder else 0)

        for _ in range(count):
            if index >= total:
                break
            try:
                raw = formatter(tok_dataset[index])
            except Exception as e:
                raise EricDatasetError(
                    f"Failed to format dataset example at index {index}: {e}"
                )

            try:
                example = {
                    k: v.tolist() if hasattr(v, "tolist") else v for k, v in raw.items()
                }
                output_file.write(json.dumps(example) + "\n")
            except Exception as e:
                raise EricIOError(
                    f"Failed to write formatted example at index {index} to file: {e}"
                )

            index += 1
