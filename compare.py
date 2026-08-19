from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

DEFAULT_PROBE = Path("data/probe-shape-swap.jsonl")
DEFAULT_V2_BATCH = Path("data/train-v2/raw/in-session-batch-v2-cross.jsonl")
DEFAULT_OUT = Path("results/delta-v1-v2.md")
# n=62 carries only 2 cross-paired rows, so it cannot show a data effect and is excluded
# from every pooled figure. It is still rendered per-N so the exclusion is visible.
POOLED_SIZES = (125, 250, 500)


class ProbeArm(StrEnum):
    """Which half of the probe a scenario belongs to, and whether v2 trained on it."""

    CONTROL = "control"
    CROSS_SEEN = "cross-seen"
    CROSS_UNSEEN = "cross-unseen"


class Rate(BaseModel):
    """A pass count over a set of trials."""

    passed: int
    total: int

    @property
    def fraction(self) -> float | None:
        """Pass rate, or None when no trials landed in this cell."""
        return self.passed / self.total if self.total else None

    def render(self) -> str:
        """Percentage with the raw counts, so a 4-trial cell cannot masquerade as precise."""
        return f"{self.fraction:.0%} ({self.passed}/{self.total})" if self.total else "-"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into dicts.

    Args:
        path: File to read.

    Returns:
        One dict per non-blank line.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"missing input: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def classify_arms(probe_path: Path, v2_batch_path: Path) -> dict[str, ProbeArm]:
    """Label every probe scenario, splitting the cross arm by what v2 actually trained on.

    The seen/unseen split is derived from the shipped v2 batch rather than hardcoded, so
    a pairing that was withheld on purpose cannot silently drift into the seen arm.

    Args:
        probe_path: The probe eval set.
        v2_batch_path: The cross-paired rows added in v2.

    Returns:
        Scenario id to arm.
    """
    trained = {
        (row["code_shape"], row["lifetime_concept"]) for row in load_jsonl(v2_batch_path)
    }
    arms: dict[str, ProbeArm] = {}
    for row in load_jsonl(probe_path):
        if row["probe_arm"] == "control":
            arms[row["id"]] = ProbeArm.CONTROL
        elif (row["code_shape"], row["lifetime_concept"]) in trained:
            arms[row["id"]] = ProbeArm.CROSS_SEEN
        else:
            arms[row["id"]] = ProbeArm.CROSS_UNSEEN
    return arms


def rate(
    trials: Sequence[dict[str, Any]],
    keep: Callable[[dict[str, Any]], bool],
) -> Rate:
    """Count judge passes among the trials matching `keep`."""
    rows = [t for t in trials if keep(t)]
    return Rate(passed=sum(1 for t in rows if t["verdict"]["passes"]), total=len(rows))


def delta(before: Rate, after: Rate) -> str:
    """Render the change in percentage points, or '-' when either side is empty."""
    if before.fraction is None or after.fraction is None:
        return "-"
    points = after.fraction - before.fraction
    return f"{points:+.0%}" if points else "0"


def render(v1: list[dict[str, Any]], v2: list[dict[str, Any]], arms: dict[str, ProbeArm],
           main_v1: list[dict[str, Any]], main_v2: list[dict[str, Any]]) -> str:
    """Build the full v1-vs-v2 comparison document."""
    sizes = sorted({t["dataset_size"] for t in v1 if t["dataset_size"]})
    lines: list[str] = [
        "# v1 vs v2 - the data change",
        "",
        "v2 = v1 plus cross-paired rows that break the shape/concept confound. Training",
        "hyperparameters and N are identical at every point, so the data is the only variable.",
        "Regenerate with `python compare.py`.",
        "",
        "## Probe: shape-swap generalisation",
        "",
        "`cross-seen` are pairings v2 trained on. `cross-unseen` were withheld from training",
        "on purpose - they are the test of whether the model learned to read the concept off",
        "the code rather than memorising four more templates.",
        "",
        "| N | control v1 | control v2 | cross-seen v1 | cross-seen v2 | cross-unseen v1 | cross-unseen v2 |",
        "| ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for size in sizes:
        cells: list[str] = []
        for arm in (ProbeArm.CONTROL, ProbeArm.CROSS_SEEN, ProbeArm.CROSS_UNSEEN):
            def keep(t: dict[str, Any], a: ProbeArm = arm, s: int = size) -> bool:
                return t["dataset_size"] == s and arms[t["scenario_id"]] is a

            before, after = rate(v1, keep), rate(v2, keep)
            cells += [before.render(), f"**{after.render()}** {delta(before, after)}"]
        lines.append(f"| {size} | " + " | ".join(cells) + " |")

    lines += [
        "",
        f"### Pooled over N={'/'.join(str(s) for s in POOLED_SIZES)}",
        "",
        "Individual cells hold 4 trials, so one judge flip moves a cell 25 points. These",
        "pooled figures are the ones to read.",
        "",
        "| Arm | v1 | v2 | delta |",
        "| --- | --- | --- | ---: |",
    ]
    for arm in (ProbeArm.CONTROL, ProbeArm.CROSS_SEEN, ProbeArm.CROSS_UNSEEN):
        def keep_pooled(t: dict[str, Any], a: ProbeArm = arm) -> bool:
            return t["dataset_size"] in POOLED_SIZES and arms[t["scenario_id"]] is a

        before, after = rate(v1, keep_pooled), rate(v2, keep_pooled)
        lines.append(f"| {arm} | {before.render()} | **{after.render()}** | {delta(before, after)} |")

    lines += [
        "",
        "## 36-scenario eval set: no-regression check",
        "",
        "| N | adherence v1 | adherence v2 | robustness v1 | robustness v2 |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for size in sorted({t["dataset_size"] for t in main_v1 if t["dataset_size"]}):
        def clean(t: dict[str, Any], s: int = size) -> bool:
            return t["dataset_size"] == s and t["category"] == "clean"

        def adversarial(t: dict[str, Any], s: int = size) -> bool:
            return t["dataset_size"] == s and t["category"] == "adversarial"

        cb, ca = rate(main_v1, clean), rate(main_v2, clean)
        rb, ra = rate(main_v1, adversarial), rate(main_v2, adversarial)
        lines.append(
            f"| {size} | {cb.render()} | **{ca.render()}** {delta(cb, ca)} "
            f"| {rb.render()} | **{ra.render()}** {delta(rb, ra)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Render the v1-vs-v2 delta tables from trial records."""
    parser = argparse.ArgumentParser(description="v1-vs-v2 data-change comparison")
    parser.add_argument("--probe-v1", type=Path, default=Path("results/probe-v1/trials.jsonl"))
    parser.add_argument("--probe-v2", type=Path, default=Path("results/probe-v2/trials.jsonl"))
    parser.add_argument("--main-v1", type=Path, default=Path("results/base-vs-tuned/trials.jsonl"))
    parser.add_argument("--main-v2", type=Path, default=Path("results/base-vs-tuned-v2/trials.jsonl"))
    parser.add_argument("--probe-set", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--v2-batch", type=Path, default=DEFAULT_V2_BATCH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    arms = classify_arms(args.probe_set, args.v2_batch)
    document = render(
        load_jsonl(args.probe_v1),
        load_jsonl(args.probe_v2),
        arms,
        load_jsonl(args.main_v1),
        load_jsonl(args.main_v2),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document)
    print(document)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
