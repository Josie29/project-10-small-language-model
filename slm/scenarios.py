from __future__ import annotations

import json
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import TypeVar, cast

from pydantic import BaseModel, Field, ValidationError, field_validator

_T = TypeVar("_T")

class Category(StrEnum):
    CLEAN = "clean"
    ADVERSARIAL = "adversarial"


class LifetimeConcept(StrEnum):
    """The single state/lifetime idea a scenario is intended to teach."""

    CREATION = "creation"
    OWNERSHIP = "ownership"
    RESET = "reset"
    ALIASING = "aliasing"


# The only keys an eval set must carry. Everything else on Scenario is a hint that
# sharpens scoring; these three are what it takes to put a prompt in front of a model at
# all. Kept as a constant because `load_scenarios` quotes it back in its error message.
REQUIRED_KEYS = ("id", "code", "student_message")


class Scenario(BaseModel):
    """One state/lifetime bug plus the student's message about it.

    Only `id`, `code`, and `student_message` are required. The remaining fields are the
    answer key our own generator emits, and they sharpen scoring when present - but a
    grader's held-out set will not have them, and refusing to load such a file would mean
    the harness cannot be run against the one eval set that most needs to run.
    """

    id: str
    code: str
    student_message: str

    category: Category = Category.CLEAN
    language: str = "python"
    bug: str | None = None
    bug_region: str | None = None
    lifetime_concept: LifetimeConcept | None = None
    expected_question_focus: str | None = None
    forbidden_fix_tokens: list[str] = Field(default_factory=list)

    @field_validator("bug", "bug_region", "expected_question_focus", mode="before")
    @classmethod
    def _blank_is_absent(cls, value: object) -> object:
        """Normalize a blank hint to an absent one.

        A blank `bug_region` is the one value that is worse than a missing one:
        `run_mechanical_check` asks whether it appears in the response, and the empty
        string appears in every string, so every trial would report a passing
        localization check. Collapsing it to None here makes that unreachable rather
        than defended separately at each call site.

        Args:
            value: The raw field value before validation.

        Returns:
            None if the value is a string that is empty or only whitespace, otherwise
            the value unchanged.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def has_rubric(self) -> bool:
        """Whether this scenario carries the answer key the judge grades against."""
        return self.bug_region is not None and self.expected_question_focus is not None


class RubricCoverage(BaseModel):
    """How much of the answer key an eval set supplies, field by field.

    A set is not simply graded or degraded - a grader could supply `bug_region` and
    nothing else. Counting per field is what lets the run manifest say which signals
    were available rather than collapsing it to one boolean.
    """

    n_scenarios: int
    with_bug: int
    with_bug_region: int
    with_expected_question_focus: int
    with_lifetime_concept: int
    with_forbidden_fix_tokens: int
    # Cannot be recovered after validation: a defaulted `clean` and an explicit `clean`
    # are the same value. Counted from the raw JSON while it is still in hand.
    defaulted_to_clean: int

    @property
    def fully_specified(self) -> bool:
        """Whether every scenario carries every scoring field."""
        return (
            self.with_bug
            == self.with_bug_region
            == self.with_expected_question_focus
            == self.with_lifetime_concept
            == self.with_forbidden_fix_tokens
            == self.n_scenarios
        )

    @property
    def degraded(self) -> bool:
        """Whether any scenario is missing a field the scoring path would have used."""
        return not self.fully_specified


def _coverage(scenarios: Sequence[Scenario], defaulted_to_clean: int) -> RubricCoverage:
    """Count answer-key coverage across a loaded scenario set."""
    return RubricCoverage(
        n_scenarios=len(scenarios),
        with_bug=sum(s.bug is not None for s in scenarios),
        with_bug_region=sum(s.bug_region is not None for s in scenarios),
        with_expected_question_focus=sum(
            s.expected_question_focus is not None for s in scenarios
        ),
        with_lifetime_concept=sum(s.lifetime_concept is not None for s in scenarios),
        with_forbidden_fix_tokens=sum(bool(s.forbidden_fix_tokens) for s in scenarios),
        defaulted_to_clean=defaulted_to_clean,
    )


def rubric_coverage(scenarios: Sequence[Scenario]) -> RubricCoverage:
    """Count answer-key coverage for an already-loaded set.

    `defaulted_to_clean` cannot be recovered once validation has run, so it is reported
    as 0 here. Prefer `load_scenarios_with_coverage`, which counts it from the raw JSON.

    Args:
        scenarios: The scenarios to describe.

    Returns:
        Per-field coverage counts.
    """
    return _coverage(scenarios, defaulted_to_clean=0)


def require_authored(value: _T | None, field: str, scenario_id: str) -> _T:
    """Narrow an optional Scenario field on a path where it is in fact mandatory.

    Scenario's answer-key fields are optional so that a held-out eval set written by
    someone else can be loaded at all. The data-authoring and training paths still
    require every one of them, and this is the boundary that says so - loudly, with the
    scenario named, rather than by widening those paths to accept None.

    Args:
        value: The optional field value.
        field: Field name, for the error message.
        scenario_id: Which scenario is missing it.

    Returns:
        The value, narrowed to non-None.

    Raises:
        ValueError: If the value is None.
    """
    if value is None:
        raise ValueError(
            f"scenario {scenario_id!r} has no {field}. Scenario's answer-key fields are "
            f"optional so a held-out eval set can be loaded, but the authoring and "
            f"training paths require the full key."
        )
    return value


def _describe_line(raw: str) -> str:
    """Summarize one JSONL line's keys for an error message."""
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError:
        return "line is not valid JSON"
    if not isinstance(parsed, dict):
        return f"line is a JSON {type(parsed).__name__}, expected an object"
    keys = ", ".join(sorted(str(k) for k in cast("dict[object, object]", parsed)))
    return f"keys present: {keys or '(none)'}"


def load_scenarios_with_coverage(path: Path) -> tuple[list[Scenario], RubricCoverage]:
    """Load scenarios from a JSONL file and describe their answer-key coverage.

    Args:
        path: Path to the scenario file.

    Returns:
        Every scenario in the file, in file order, plus per-field coverage counts.

    Raises:
        ValueError: If the file contains no scenarios, is a JSON array rather than
            JSONL, or has a line that does not validate. The message names the line
            number and the keys that line actually carried.
    """
    text = path.read_text()
    if text.lstrip().startswith("["):
        raise ValueError(
            f"{path} looks like a JSON array. This harness reads JSONL - one JSON "
            f"object per line, no enclosing brackets or commas."
        )

    scenarios: list[Scenario] = []
    defaulted_to_clean = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            scenario = Scenario.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(
                f"{path}:{lineno} is not a usable scenario. Required keys are "
                f"{', '.join(REQUIRED_KEYS)}; {_describe_line(line)}.\n{exc}"
            ) from exc
        scenarios.append(scenario)
        if '"category"' not in line:
            defaulted_to_clean += 1

    if not scenarios:
        raise ValueError(f"no scenarios in {path}")
    return scenarios, _coverage(scenarios, defaulted_to_clean)


def load_scenarios(path: Path) -> list[Scenario]:
    """Load scenarios from a JSONL file.

    Args:
        path: Path to the scenario file.

    Returns:
        Every scenario in the file, in file order.

    Raises:
        ValueError: If the file contains no scenarios or a line does not validate.
    """
    scenarios, _ = load_scenarios_with_coverage(path)
    return scenarios


def stratified_sample(scenarios: Sequence[Scenario], limit: int) -> list[Scenario]:
    """Take `limit` scenarios split across categories.

    The file is ordered clean-first, so a naive head-slice yields an all-clean sample
    and never exercises the adversarial scenarios the thesis rests on.

    An even split is only possible when every category has enough rows. A 36-scenario
    set of 24 clean and 12 adversarial cannot give 18 of each, and a held-out set may
    carry no category labels at all and land entirely in one bucket. Both cases top up
    in file order rather than silently returning fewer scenarios than asked for.

    Args:
        scenarios: The full scenario set.
        limit: Total number of scenarios wanted.

    Returns:
        Up to `limit` scenarios, split as evenly as the set allows. Fewer only when the
        set itself is smaller than `limit`.
    """
    per_category = max(1, limit // len(Category))
    picked: list[Scenario] = []
    for category in Category:
        picked += [s for s in scenarios if s.category is category][:per_category]

    if len(picked) < limit:
        already = {id(s) for s in picked}
        picked += [s for s in scenarios if id(s) not in already][: limit - len(picked)]
    return picked[:limit] if len(picked) > limit else picked
