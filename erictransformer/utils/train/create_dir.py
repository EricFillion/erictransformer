import datetime as dt
import logging
import os
import re
from pathlib import Path

from erictransformer.exceptions import EricIOError


def _slugify(name: str) -> str:
    safe_chars = re.compile(r"[^A-Za-z0-9._-]+")
    s = name.strip()
    s = s.replace(" ", "-")
    s = safe_chars.sub("-", s)
    s = re.sub(r"-{2,}", "-", s)
    s = s.strip("-._")
    if not s:
        raise EricIOError(f"Invalid directory name: {name}")
    return s


def _increment_path(base_path: Path) -> Path:
    parent = base_path.parent
    name = base_path.name

    m = re.match(r"^(.*?)(?:-(\d+))?$", name)
    root = m.group(1)
    pat = re.compile(rf"^{re.escape(root)}(?:-(\d+))?$")

    max_i = -1
    if parent.exists():
        for child in parent.iterdir():
            m2 = pat.match(child.name)
            if m2:
                i = int(m2.group(1)) if m2 and m2.group(1) else 0
                if i > max_i:
                    max_i = i

    if max_i < 0:
        return base_path
    next_i = max_i + 1
    if next_i == 0:
        return root
    else:
        return parent / f"{name}-{next_i}"


def make_dir(name):
    try:
        os.makedirs(name, exist_ok=True)
        logging.info(f"Directory {name} created successfully.")
    except FileExistsError:
        logging.warning(f"Directory {name} already exists.")


def create_tracker_dir(out_dir: str, label: str, run_name: str):
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        slugify_run_name = _slugify(run_name)

        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)

        dir_name = _increment_path(output / slugify_run_name)

        make_dir(dir_name)

    else:
        dir_name = os.path.join(out_dir, f"{label}_{timestamp}")
        make_dir(dir_name)

    return dir_name
