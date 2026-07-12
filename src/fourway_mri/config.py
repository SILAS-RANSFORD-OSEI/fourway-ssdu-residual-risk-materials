from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    return cfg


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path:
        Directory path.

    Returns
    -------
    pathlib.Path
        Resolved directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
