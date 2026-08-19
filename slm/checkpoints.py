from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

DEFAULT_CHECKPOINTS = Path("results/checkpoints.jsonl")

# train.py names every repo `<user>/qwen3-0.6b-state-lifetime-tutor-n<size>`, so the curve
# point is recoverable from the id alone when the manifest is missing - which is the case
# for a grader who cloned the repo and only has the Hub to go on.
#
# Trailing segments after the size are tolerated so that a dataset revision trained with
# `train.py --repo-suffix` (`...-n125-v2`) still reports its curve point. Without this the
# size comes back None and every v2 checkpoint drops out of the curve table.
_SIZE_SUFFIX = re.compile(r"-n(\d+)(?:-[A-Za-z0-9.]+)*$")


class Checkpoint(BaseModel):
    """One trained checkpoint, and everything needed to pin a run against it."""

    repo_id: str
    dataset_size: int
    base_model: str
    epochs: int
    # Hub commit sha. None until the push completes, so a failed upload is visible rather
    # than silently recorded as a reproducible artifact.
    revision: str | None = None
    train_loss: float | None = None


def infer_dataset_size(repo_id: str) -> int | None:
    """Recover the curve point from a repo id.

    Args:
        repo_id: Hugging Face repo id or local path.

    Returns:
        The training-set size encoded in a trailing `-n<int>`, or None if absent.
    """
    match = _SIZE_SUFFIX.search(repo_id)
    return int(match.group(1)) if match else None


def load_checkpoints(path: Path = DEFAULT_CHECKPOINTS) -> list[Checkpoint]:
    """Load the checkpoint manifest, or return an empty list if it does not exist yet."""
    if not path.exists():
        return []
    return [
        Checkpoint.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def record_checkpoint(
    checkpoint: Checkpoint, path: Path = DEFAULT_CHECKPOINTS
) -> list[Checkpoint]:
    """Upsert one checkpoint into the manifest, keyed by repo id.

    Retraining a curve point replaces its row rather than appending a second one, so the
    manifest always describes the checkpoints currently on the Hub.

    Args:
        checkpoint: The record to write.
        path: Manifest location.

    Returns:
        The manifest after the write, ordered by dataset size.
    """
    existing = [c for c in load_checkpoints(path) if c.repo_id != checkpoint.repo_id]
    merged = sorted([*existing, checkpoint], key=lambda c: c.dataset_size)
    write_checkpoints(merged, path)
    return merged


def write_checkpoints(
    checkpoints: Sequence[Checkpoint], path: Path = DEFAULT_CHECKPOINTS
) -> None:
    """Write the checkpoint manifest as JSONL, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{c.model_dump_json()}\n" for c in checkpoints))
