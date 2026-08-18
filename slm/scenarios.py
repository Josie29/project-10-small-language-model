from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

class Category(StrEnum):
    CLEAN = "clean"
    ADVERSARIAL = "adversarial"

class Scenario(BaseModel):
    """One test case: buggy code plus the message the student sends about it."""

    id: str
    category: Category
    language: str
    code: str
    student_message: str
    bug: str
    forbidden_fix_tokens: list[str]

def load_scenarios(path: Path) -> list[Scenario]:
    """Load scenarios from a JSONL file.

    Args:
        path: Path to the scenario file.

    Returns:
        Every scenario in the file, in file order.

    Raises:
        ValueError: If the file contains no scenarios.
    """
    scenarios = [
        Scenario.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not scenarios:
        raise ValueError(f"no scenarios in {path}")
    return scenarios


def stratified_sample(scenarios: Sequence[Scenario], limit: int) -> list[Scenario]:
    """Take `limit` scenarios split across categories.

    The file is ordered clean-first, so a naive head-slice yields an all-clean sample
    and never exercises the adversarial scenarios the thesis rests on.

    Args:
        scenarios: The full scenario set.
        limit: Total number of scenarios wanted.

    Returns:
        Up to `limit` scenarios, split as evenly as possible across categories.
    """
    per_category = max(1, limit // len(Category))
    picked: list[Scenario] = []
    for category in Category:
        picked += [s for s in scenarios if s.category is category][:per_category]
    return picked[:limit] if len(picked) > limit else picked
