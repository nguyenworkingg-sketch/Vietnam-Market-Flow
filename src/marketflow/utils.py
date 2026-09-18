from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dirs(root: str | Path) -> None:
    root = Path(root)
    for p in [root/'data'/'raw', root/'data'/'processed', root/'outputs']:
        p.mkdir(parents=True, exist_ok=True)
