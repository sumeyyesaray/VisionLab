"""Collects the "what produced this" facts a checkpoint or MLflow run needs
to be self-describing: the exact code (git commit), the exact config
(content hash, not just a filename that can change), and the exact data
(DVC's content hash for the dataset directory, when the dataset is
DVC-tracked — see dvc.yaml). Used by both src/models/checkpoint.py
(embedded in the .pt file itself) and src/training/tracking.py (logged as
MLflow params/tags), so both stay in sync from one source of truth.
"""

import hashlib
import subprocess
from pathlib import Path

import yaml


def _run_git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=5
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def get_git_info() -> dict:
    commit = _run_git("rev-parse", "HEAD")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = _run_git("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": bool(dirty) if dirty is not None else None,
    }


def get_config_hash(config_path: str | Path) -> str | None:
    """MD5 of the config file's raw bytes — catches edits to a config that
    keeps the same filename across runs, which a path string alone would
    miss."""
    path = Path(config_path)
    if not path.exists():
        return None
    return hashlib.md5(path.read_bytes()).hexdigest()


def get_dvc_data_version(dataset_root: str | Path) -> str | None:
    """Reads the `.dvc` pointer file's content hash for a DVC-tracked
    dataset directory (e.g. datasets/extracted/mushroom.dvc), so a
    checkpoint/run records exactly which version of the data it trained on.
    Returns None if the path isn't DVC-tracked (e.g. PlantVillage, which has
    no pipeline yet)."""
    dvc_file = Path(str(dataset_root).rstrip("/") + ".dvc")
    if not dvc_file.exists():
        return None
    try:
        with open(dvc_file, encoding="utf-8") as f:
            spec = yaml.safe_load(f)
        return spec["outs"][0]["md5"]
    except (yaml.YAMLError, KeyError, IndexError, OSError):
        return None


def collect_run_metadata(config_path: str | Path, dataset_root: str | Path | None = None) -> dict:
    """The full self-describing bundle: code version + config version + data
    version. Safe to call outside a git repo or without DVC set up — fields
    just come back as None rather than raising."""
    metadata = {
        "config_path": str(config_path),
        "config_hash": get_config_hash(config_path),
        **get_git_info(),
    }
    if dataset_root is not None:
        metadata["data_version"] = get_dvc_data_version(dataset_root)
    return metadata
