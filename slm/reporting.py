from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from slm.checks import MechanicalCheck
from slm.config import Family
from slm.judge import JudgeVerdict
from slm.prompting import Strategy
from slm.scenarios import Category, LifetimeConcept, RubricCoverage

class Trial(BaseModel):
    """One cell of the sweep applied to one scenario."""

    scenario_id: str
    category: Category
    # None when the eval set did not label the scenario with a concept, which is the
    # normal case for a held-out set written by someone else. Optional for the same
    # reason as `dataset_size` below: absence is a fact about the input, not a gap to
    # fill in with a guess.
    lifetime_concept: LifetimeConcept | None = None
    model_id: str
    family: Family
    strategy: Strategy
    response: str
    check: MechanicalCheck
    verdict: JudgeVerdict
    # How many training examples produced this checkpoint. None for the prompt-ceiling
    # ablation and for the untuned base, neither of which sits on the curve. Optional so
    # trials.jsonl written before the fine-tuning phase still parses.
    dataset_size: int | None = None


class CellResult(BaseModel):
    """Aggregate scores for one model x strategy combination."""

    model_id: str
    strategy: Strategy
    spec_adherence: float
    robustness: float
    mechanical_pass_rate: float
    n_clean: int
    n_adversarial: int
    dataset_size: int | None = None
    # How many scenarios this cell should have scored. `run_trial` drops a scenario whose
    # generation or judge call failed, which shrinks the denominator with no trace in the
    # table - a confident wrong number. None when the caller did not say.
    n_expected: int | None = None

def _rate(trials: Sequence[Trial], category: Category) -> tuple[float, int]:
    """Return the judge pass rate and sample count for one scenario category."""
    subset = [t for t in trials if t.category is category]
    if not subset:
        return 0.0, 0
    return sum(t.verdict.passes for t in subset) / len(subset), len(subset)


def aggregate(
    trials: Sequence[Trial], n_scenarios: int | None = None
) -> list[CellResult]:
    """Collapse trials into one row per model x strategy cell.

    Args:
        trials: All completed trials.
        n_scenarios: Size of the eval set, when known, so a cell can report scenarios it
            failed to score rather than quietly shrinking its own denominator.

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
                    dataset_size=subset[0].dataset_size,
                    n_expected=n_scenarios,
                )
            )
    return cells


def render_degraded_note(coverage: RubricCoverage) -> str:
    """Describe, in one paragraph, what a partially-specified eval set could not score.

    Kept next to the renderer because `table.md` is pasted into decks without `run.json`
    beside it. A number whose grounding lives in another file is a number that will be
    quoted without it.

    Args:
        coverage: Per-field answer-key counts for the eval set.

    Returns:
        A Markdown paragraph, or an empty string when the set is fully specified.
    """
    if not coverage.degraded:
        return ""
    n = coverage.n_scenarios
    missing_rubric = n - min(coverage.with_bug_region, coverage.with_expected_question_focus)
    parts = [
        f"**Scored in degraded mode.** {missing_rubric} of {n} scenarios supplied no "
        "`bug_region` or `expected_question_focus`, so the judge graded them against the "
        "behavior spec alone and identified the bug from its own reading of the code. "
        "The mechanical clauses those fields feed - `has_localization` and `stated_fix` "
        "- had nothing to check against and dropped out of the mechanical pass rate.",
        "Dropping a check can only remove evidence a response could have *failed* on, "
        "never evidence it could have passed on, so these numbers are biased high and "
        "are not comparable to any pinned run in `results/`.",
        "The behavior spec is scoped to four state/lifetime concepts (`slm/spec.py`, "
        "edge case 6). A scenario outside that scope scores the spec, not the model.",
    ]
    if coverage.defaulted_to_clean:
        parts.insert(
            1,
            f"{coverage.defaulted_to_clean} scenarios carried no `category` and were "
            "counted as clean. Spec adherence is measured over them; Robustness is "
            "measured only over scenarios explicitly labelled adversarial. Neither "
            "number is a claim about which they actually are.",
        )
    return "\n\n".join(parts)


def render_table(
    cells: Sequence[CellResult],
    trials: Sequence[Trial],
    title: str = "Prompt-Ceiling Ablation — Results",
    coverage: RubricCoverage | None = None,
) -> str:
    """Render the results table and violation breakdown as Markdown.

    Only the heading is configurable. The columns, metrics, and violation vocabulary are
    shared with the ablation on purpose: base-vs-tuned numbers are only comparable to the
    prompt ceiling if they are computed and presented the same way.

    Args:
        cells: Aggregated per-cell scores.
        trials: All trials, used for the violation-frequency breakdown.
        title: Heading for the document.

    Returns:
        A Markdown document ready to paste into the defense deck.
    """
    lines = [
        f"# {title}",
        "",
        "Spec adherence = judge pass rate on clean scenarios.",
        "Robustness = judge pass rate on adversarial scenarios.",
    ]
    note = render_degraded_note(coverage) if coverage is not None else ""
    if note:
        lines += ["", note]
    lines += [
        "",
        "| Model | Strategy | Spec adherence | Robustness | Mechanical pass | n |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for c in cells:
        scored = c.n_clean + c.n_adversarial
        # Scenarios that failed to run are shown in the cell, not a footnote: a bare "34"
        # against a 36-scenario set reads as a complete run.
        n = (
            f"{scored}/{c.n_expected}"
            if c.n_expected is not None and scored != c.n_expected
            else str(scored)
        )
        # A category with no samples is not a zero score - say so rather than imply one.
        adherence = f"{c.spec_adherence:.0%}" if c.n_clean else "--"
        robustness = f"{c.robustness:.0%}" if c.n_adversarial else "--"
        lines.append(
            f"| `{c.model_id}` | {c.strategy} | {adherence} | "
            f"{robustness} | {c.mechanical_pass_rate:.0%} | {n} |"
        )

    lines += ["", "## Judge pass rate by state/lifetime concept", ""]
    if any(t.lifetime_concept is not None for t in trials):
        lines += [
            "| Model | Strategy | Concept | Pass rate | n |",
            "| --- | --- | --- | ---: | ---: |",
        ]
        # None is appended after the enum members so an unlabelled row cannot be mistaken
        # for a concept, and so a set that mixes the two still shows both.
        concepts: list[LifetimeConcept | None] = [*LifetimeConcept, None]
        for model_id in dict.fromkeys(t.model_id for t in trials):
            for strategy in Strategy:
                for concept in concepts:
                    subset = [
                        t
                        for t in trials
                        if t.model_id == model_id
                        and t.strategy is strategy
                        and t.lifetime_concept is concept
                    ]
                    if subset:
                        pass_rate = sum(t.verdict.passes for t in subset) / len(subset)
                        label = concept if concept is not None else "(unlabelled)"
                        lines.append(
                            f"| `{model_id}` | {strategy} | {label} | "
                            f"{pass_rate:.0%} | {len(subset)} |"
                        )
    else:
        # A breakdown with nothing to break down by is not an empty table, it is an
        # absent one. Saying so beats printing headers a grader would read as a result.
        lines.append(
            "Not available: no scenario in this eval set carried a `lifetime_concept`, "
            "so there is nothing to break the pass rate down by. The table above is the "
            "whole story."
        )

    violations: dict[str, int] = {}
    for t in trials:
        if not t.verdict.passes and t.verdict.violation:
            violations[t.verdict.violation] = violations.get(t.verdict.violation, 0) + 1
    lines += ["", "## Violations across all cells", "", "| Violation | Count |", "| --- | ---: |"]
    for name, count in sorted(violations.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {count} |")
    return "\n".join(lines) + "\n"


def render_curve(cells: Sequence[CellResult]) -> str:
    """Render the data-efficiency curve as Markdown.

    Only cells carrying a `dataset_size` appear, so the untuned base and any frontier rows
    are excluded rather than plotted at an imaginary N.

    Args:
        cells: Aggregated per-cell scores.

    Returns:
        A Markdown document, or an empty string when no cell sits on the curve.
    """
    points = sorted(
        (c for c in cells if c.dataset_size is not None),
        key=lambda c: c.dataset_size or 0,
    )
    if not points:
        return ""
    lines = [
        "# Data-Efficiency Curve",
        "",
        "Spec adherence = judge pass rate on clean scenarios.",
        "Robustness = judge pass rate on adversarial scenarios.",
        "",
        "Epochs are held fixed across every point, so the smallest N also trains for the",
        "fewest optimizer steps. Read the low end as data *and* step starvation together.",
        "",
        "| N | Checkpoint | Spec adherence | Robustness | Mechanical pass |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for c in points:
        adherence = f"{c.spec_adherence:.0%}" if c.n_clean else "--"
        robustness = f"{c.robustness:.0%}" if c.n_adversarial else "--"
        lines.append(
            f"| {c.dataset_size} | `{c.model_id}` | {adherence} | "
            f"{robustness} | {c.mechanical_pass_rate:.0%} |"
        )
    return "\n".join(lines) + "\n"


def write_results(
    trials: Sequence[Trial],
    out_dir: Path,
    title: str = "Prompt-Ceiling Ablation — Results",
    *,
    coverage: RubricCoverage | None = None,
    n_scenarios: int | None = None,
) -> None:
    """Write raw trials and the rendered results table to disk.

    Writes `curve.md` as well when any trial carries a dataset size, so the ablation's
    output is byte-for-byte what it always was.

    Args:
        trials: Every completed trial.
        out_dir: Directory to create and write into.
        title: Heading for the results table.
        coverage: Answer-key coverage of the eval set. When it shows anything missing,
            the table carries a note saying what could not be scored.
        n_scenarios: Size of the eval set, so cells can report scenarios they dropped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.jsonl"
    trials_path.write_text("\n".join(t.model_dump_json() for t in trials) + "\n")
    cells = aggregate(trials, n_scenarios)
    table = render_table(cells, trials, title, coverage)
    (out_dir / "table.md").write_text(table)
    print(f"\n{len(trials)} trials -> {trials_path}")
    print(table)

    curve = render_curve(cells)
    if curve:
        (out_dir / "curve.md").write_text(curve)
        print(curve)
