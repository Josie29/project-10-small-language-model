from __future__ import annotations

import re

from pydantic import BaseModel

from slm.scenarios import Scenario

class MechanicalCheck(BaseModel):
    """Deterministic spec checks that need no model call."""

    emitted_code: bool
    stated_fix: bool
    question_count: int
    has_localization: bool
    possible_compound_question: bool

    @property
    def passed(self) -> bool:
        """True when no mechanical violation was found."""
        return (
            not self.emitted_code
            and not self.stated_fix
            and self.question_count == 1
            and self.has_localization
            and not self.possible_compound_question
        )

_CODE_FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_CODE_LIKE_LINE = re.compile(
    r"^\s*(?:(?:def|class|for|while|if|elif|else|return|import|from)\b|"
    r"[A-Za-z_][\w.\[\]]*\s*(?:=|\(|\[))"
)
_QUESTION_WORD = re.compile(r"\b(?:what|when|where|which|who|why|how)\b", re.IGNORECASE)
_QUESTION_CONJUNCTION = re.compile(r"\b(?:and|or)\b", re.IGNORECASE)


def _normalize(text: str) -> str:
    """Collapse all whitespace runs to single spaces for substring comparison."""
    return " ".join(text.split())


def normalize_token(text: str) -> str:
    """Normalize prose while tolerating spacing differences in code fragments."""
    return re.sub(r"\s*([=()\[\]{},.:])\s*", r"\1", _normalize(text)).lower()


def _contains_new_code(response: str, student_code: str) -> bool:
    """Return whether a response contains non-verbatim code in common output forms."""
    quoted_blocks = _CODE_FENCE.findall(response)
    inline_fragments = _INLINE_CODE.findall(response)
    if any(_normalize(fragment) not in student_code for fragment in [*quoted_blocks, *inline_fragments]):
        return True

    for line in response.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or not _CODE_LIKE_LINE.match(line):
            continue
        if _normalize(stripped) not in student_code:
            return True
    return False


def _question_text(response: str) -> str:
    """Return the final question clause for a deliberately conservative diagnostic."""
    before_mark = response.rsplit("?", maxsplit=1)[0]
    return re.split(r"[.!\n]", before_mark)[-1]


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
    question = _question_text(response) if response.count("?") == 1 else ""
    normalized_response = normalize_token(response)
    stated_fix = any(
        normalize_token(token) in normalized_response
        for token in scenario.forbidden_fix_tokens
    )
    return MechanicalCheck(
        emitted_code=_contains_new_code(response, student_code),
        stated_fix=stated_fix,
        question_count=response.count("?"),
        has_localization=_normalize(scenario.bug_region) in _normalize(response),
        possible_compound_question=(
            len(_QUESTION_WORD.findall(question)) > 1
            or bool(_QUESTION_CONJUNCTION.search(question))
        ),
    )
