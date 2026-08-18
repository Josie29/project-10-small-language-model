from __future__ import annotations

import argparse
import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Any

from slm.checkpoints import DEFAULT_CHECKPOINTS, Checkpoint, record_checkpoint
from slm.config import BASE_MODEL, load_env_file
from slm.dataset import load_pool
from slm.publishing import push_checkpoint
from slm.sft import run_sft
from slm.training import EPOCHS, TrainConfig, TrainResult, curve_subset, sft_rows

DEFAULT_POOL = Path("data/train/pool-v1.jsonl")
DEFAULT_LOGS = Path("results/train")
DEFAULT_WEIGHTS = Path("checkpoints")
REPO_PREFIX = "qwen3-0.6b-state-lifetime-tutor"


class Backend(StrEnum):
    """Where the training run executes.

    The recipe is identical either way; only the frozen base's precision differs. See the
    QLoRA note in the module docstring of `slm/sft.py`.
    """

    LOCAL = "local"
    MODAL = "modal"


def parse_sizes(raw: str) -> list[int]:
    """Parse a comma-separated list of curve-point sizes.

    Args:
        raw: e.g. "62,125,250,500".

    Returns:
        The sizes, ascending, deduplicated.

    Raises:
        ValueError: If the list is empty or any entry is not a positive integer.
    """
    sizes = {int(part) for part in raw.split(",") if part.strip()}
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError(f"invalid --sizes: {raw!r}")
    return sorted(sizes)


def write_run_logs(result: TrainResult, logs_dir: Path) -> Path:
    """Write one run's training logs where a grader can read them without an account.

    The brief requires inspectable training logs; committing them to git rather than a
    hosted tracker keeps the reproduction path free of a second signup.

    Args:
        result: The finished run.
        logs_dir: Parent directory for per-run subdirectories.

    Returns:
        The directory written.
    """
    run_dir = logs_dir / f"n-{result.dataset_size}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        result.model_dump_json(indent=2, exclude={"trainer_state"}) + "\n"
    )
    (run_dir / "trainer_state.json").write_text(
        json.dumps(result.trainer_state, indent=2) + "\n"
    )
    (run_dir / "loss.csv").write_text(
        "step,epoch,loss\n"
        + "".join(f"{p.step},{p.epoch},{p.loss}\n" for p in result.loss_curve)
    )
    return run_dir


def train_remote(config: TrainConfig, rows: list[dict[str, Any]]) -> TrainResult:
    """Run one curve point on a Modal A10G with a true 4-bit base.

    Kept alongside the local path so the same recipe can be re-run as literal QLoRA once
    GPU billing is available, without editing anything but the backend flag. The remote
    side pushes its own weights, because they never exist on this machine.

    Args:
        config: Hyperparameters and destination repo.
        rows: Chat-format training rows.

    Returns:
        The finished run, with `revision` already set.
    """
    import modal

    from modal_app import app, train_one

    with modal.enable_output(), app.run():
        return train_one.remote(config, rows)


def main() -> None:
    """Train one checkpoint per curve point and push each to the Hub."""
    parser = argparse.ArgumentParser(description="QLoRA/LoRA training sweep")
    parser.add_argument(
        "--sizes", default="500", help="Comma-separated curve points, e.g. 62,125,250,500"
    )
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--backend", type=Backend, choices=list(Backend), default=Backend.LOCAL)
    parser.add_argument("--device", default=None, help="torch device; autodetected when omitted")
    parser.add_argument("--hf-user", default=None, help="Hub namespace; defaults to $HF_USER")
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument(
        "--no-push", action="store_true", help="Train and log, but leave the weights local"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Slice the pool and render configs without loading weights or touching the Hub",
    )
    args = parser.parse_args()

    load_env_file()
    sizes = parse_sizes(args.sizes)
    pool = load_pool(args.pool)
    if not pool:
        raise SystemExit(f"no training examples in {args.pool}")

    hf_user = args.hf_user or os.environ.get("HF_USER")
    if not hf_user and not args.dry_run:
        raise SystemExit("set HF_USER in .env or pass --hf-user")

    jobs: list[tuple[TrainConfig, list[dict[str, Any]]]] = []
    for size in sizes:
        config = TrainConfig(
            base_model=args.base,
            dataset_size=size,
            repo_id=f"{hf_user or 'DRY'}/{REPO_PREFIX}-n{size}",
            epochs=args.epochs,
        )
        jobs.append((config, sft_rows(curve_subset(pool, size))))

    print(f"Pool: {len(pool)} examples from {args.pool}  |  backend: {args.backend}")
    for config, rows in jobs:
        print(
            f"  n={config.dataset_size:<4} rows={len(rows):<4} "
            f"steps~{config.expected_steps:<4} -> {config.repo_id}"
        )

    if args.dry_run:
        print(f"\nDRY RUN - first row:\n{jobs[0][1][0]}")
        return

    for config, rows in jobs:
        print(f"\n=== training n={config.dataset_size} ===")
        output_dir = args.weights / f"n-{config.dataset_size}"
        if args.backend is Backend.MODAL:
            result = train_remote(config, rows)
        else:
            result = run_sft(config, rows, output_dir, device=args.device)

        # The Modal path pushes from inside the container, so its revision is already set.
        if not args.no_push and result.revision is None:
            result.revision = push_checkpoint(result)
        run_dir = write_run_logs(result, args.logs)
        record_checkpoint(
            Checkpoint(
                repo_id=result.repo_id,
                dataset_size=result.dataset_size,
                base_model=config.base_model,
                epochs=config.epochs,
                revision=result.revision,
                train_loss=result.train_loss,
            ),
            args.checkpoints,
        )
        print(
            f"{result.repo_id}@{result.revision or 'local-only'}  "
            f"loss={result.train_loss:.4f}  steps={result.steps}  "
            f"{result.runtime_seconds:.0f}s  logs -> {run_dir}"
        )


if __name__ == "__main__":
    main()
