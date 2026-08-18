from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from slm.checks import MechanicalCheck
from slm.config import Family
from slm.judge import JudgeVerdict
from slm.prompting import Strategy
from slm.scenarios import Category

class Trial(BaseModel):
    """One cell of the sweep applied to one scenario."""

    scenario_id: str
    category: Category
    model_id: str
    family: Family
    strategy: Strategy
    response: str
    check: MechanicalCheck
    verdict: JudgeVerdict


class CellResult(BaseModel):
    """Aggregate scores for one model x strategy combination."""

    model_id: str
    strategy: Strategy
    spec_adherence: float
    robustness: float
    mechanical_pass_rate: float
    n_clean: int
    n_adversarial: int

def _rate(trials: Sequence[Trial], category: Category) -> tuple[float, int]:
    """Return the judge pass rate and sample count for one scenario category."""
    subset = [t for t in trials if t.category is category]
    if not subset:
        return 0.0, 0
    return sum(t.verdict.passes for t in subset) / len(subset), len(subset)


def aggregate(trials: Sequence[Trial]) -> list[CellResult]:
    """Collapse trials into one row per model x strategy cell.

    Args:
        trials: All completed trials.

    Returns:
        One CellResult per cell, ordered by model then strategy.
    """
    cells: list[CellResult] = []
    for model_id in dict.fromkeys(t.model_id for t in trials):
        for strategy in Strategy:
            subset = [
                t for t in trials if t.model_id == model_id and t.strategy is strategy
            ]
            if not subset:
                continue
            adherence, n_clean = _rate(subset, Category.CLEAN)
            robustness, n_adversarial = _rate(subset, Category.ADVERSARIAL)
            cells.append(
                CellResult(
                    model_id=model_id,
                    strategy=strategy,
                    spec_adherence=adherence,
                    robustness=robustness,
                    mechanical_pass_rate=sum(t.check.passed for t in subset)
                    / len(subset),
                    n_clean=n_clean,
                    n_adversarial=n_adversarial,
                )
            )
    return cells


def render_table(cells: Sequence[CellResult], trials: Sequence[Trial]) -> str:
    """Render the results table and violation breakdown as Markdown.

    Args:
        cells: Aggregated per-cell scores.
        trials: All trials, used for the violation-frequency breakdown.

    Returns:
        A Markdown document ready to paste into the defense deck.
    """
    lines = [
        "# Prompt-Ceiling Ablation — Results",
        "",
        "Spec adherence = judge pass rate on clean scenarios.",
        "Robustness = judge pass rate on adversarial scenarios.",
        "",
        "| Model | Strategy | Spec adherence | Robustness | Mechanical pass | n |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for c in cells:
        n = c.n_clean + c.n_adversarial
        # A category with no samples is not a zero score - say so rather than imply one.
        adherence = f"{c.spec_adherence:.0%}" if c.n_clean else "--"
        robustness = f"{c.robustness:.0%}" if c.n_adversarial else "--"
        lines.append(
            f"| `{c.model_id}` | {c.strategy} | {adherence} | "
            f"{robustness} | {c.mechanical_pass_rate:.0%} | {n} |"
        )

    violations: dict[str, int] = {}
    for t in trials:
        if not t.verdict.passes and t.verdict.violation:
            violations[t.verdict.violation] = violations.get(t.verdict.violation, 0) + 1
    lines += ["", "## Violations across all cells", "", "| Violation | Count |", "| --- | ---: |"]
    for name, count in sorted(violations.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {count} |")
    return "\n".join(lines) + "\n"


def write_results(trials: Sequence[Trial], out_dir: Path) -> None:
    """Write raw trials and the rendered results table to disk.

    Args:
        trials: Every completed trial.
        out_dir: Directory to create and write into.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.jsonl"
    trials_path.write_text("\n".join(t.model_dump_json() for t in trials) + "\n")
    table = render_table(aggregate(trials), trials)
    (out_dir / "table.md").write_text(table)
    print(f"\n{len(trials)} trials -> {trials_path}")
    print(table)
