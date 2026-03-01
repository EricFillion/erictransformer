import pathlib

import huggingface_hub


def load_from_repo_or_path(path: str, path_in_repo: str) -> str:
    # Resolve a path that is either local or a repo into a local path.
    if pathlib.Path(path).exists():
        return path
    else:
        local_repo_dir = huggingface_hub.snapshot_download(
            repo_id=path, repo_type="dataset", token=True
        )
        return str(pathlib.Path(local_repo_dir) / path_in_repo)
