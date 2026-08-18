from __future__ import annotations

import re

from pydantic import BaseModel

from slm.scenarios import Scenario

class MechanicalCheck(BaseModel):
    """Deterministic spec checks that need no model call."""

    emitted_code: bool
    stated_fix: bool
    question_count: int

    @property
    def passed(self) -> bool:
        """True when no mechanical violation was found."""
        return not self.emitted_code and not self.stated_fix and self.question_count == 1

_CODE_FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)


def _normalize(text: str) -> str:
    """Collapse all whitespace runs to single spaces for substring comparison."""
    return " ".join(text.split())


def run_mechanical_check(response: str, scenario: Scenario) -> MechanicalCheck:
    """Score one response against the spec's mechanically-checkable clauses.

    Args:
        response: The model's response text.
        scenario: The scenario it was responding to, supplying the student's code
            (quoting it back is allowed) and the known fix tokens.

    Returns:
        The three check results. See MechanicalCheck.passed for the verdict.
    """
    student_code = _normalize(scenario.code)
    emitted_code = any(
        _normalize(block) not in student_code
        for block in _CODE_FENCE.findall(response)
    )
    lowered = response.lower()
    stated_fix = any(token.lower() in lowered for token in scenario.forbidden_fix_tokens)
    return MechanicalCheck(
        emitted_code=emitted_code,
        stated_fix=stated_fix,
        question_count=response.count("?"),
    )
