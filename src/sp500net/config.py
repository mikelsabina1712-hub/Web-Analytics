"""Configuration and paths.

Every parameter lives in ``config.yaml`` at the repository root, so a
notebook never hard-codes a date, threshold or seed.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Read ``config.yaml`` (defaults to the one at the repository root)."""
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def in_colab() -> bool:
    """True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def data_dir(cfg: dict, create: bool = True) -> Path:
    """Folder for raw and derived data (Drive in Colab, ./data locally).

    Sub-folders:
      raw/        downloads exactly as received (prices, Wikipedia HTML)
      reference/  small tables built from raw data (ticker map, revisions)
      interim/    cleaned data passed between notebooks
    """
    if in_colab():
        root = Path(cfg["paths"]["drive_data_dir"])
    else:
        root = REPO_ROOT / cfg["paths"]["local_data_dir"]
    if create:
        for sub in ("raw", "reference", "interim"):
            (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def get_secret(name: str) -> str | None:
    """Read a secret from Colab Secrets, falling back to an environment variable.

    The value is returned, never printed. Returns None if it is not set.
    """
    if in_colab():
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception:  # secret missing or notebook access not granted
            pass
    return os.environ.get(name)
