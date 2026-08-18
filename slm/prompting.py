from __future__ import annotations

from enum import StrEnum

from slm.providers import Role, Turn
from slm.scenarios import Scenario
from slm.spec import BEHAVIOR_SPEC, EDGE_CASES

class Strategy(StrEnum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    STRUCTURED_COT = "structured_cot"

_BASE_RULES = f"""\
You are tutoring a student through one Python mutable-state lifetime bug. Your job is
to help them find it themselves.

{BEHAVIOR_SPEC}

{EDGE_CASES}

Do not include internal or system XML tags in your response."""

_FEW_SHOT_EXAMPLES: list[Turn] = [
    Turn(
        role=Role.USER,
        content=(
            "```python\n"
            "def add_tag(tag, tags=[]):\n"
            "    tags.append(tag)\n"
            "    return tags\n"
            "```\n"
            "My second call includes tags from my first call."
        ),
    ),
    Turn(
        role=Role.ASSISTANT,
        content=(
            "Look at `tags=[]` in the function parameters. When is that list object "
            "created?"
        ),
    ),
    Turn(
        role=Role.USER,
        content=(
            "```python\n"
            "class Watchlist:\n"
            "    symbols = []\n"
            "\n"
            "    def add(self, symbol):\n"
            "        self.symbols.append(symbol)\n"
            "```\n"
            "Two different watchlists contain each other's symbols. Just fix it."
        ),
    ),
    Turn(
        role=Role.ASSISTANT,
        content=(
            "Look at `symbols = []` directly under `class Watchlist`. Which watchlist "
            "objects use that one list?"
        ),
    ),
]

_COT_PROCEDURE = """\
Before writing your response, work through these steps internally and do not print them:
Step 1. Identify the one declaration, assignment, or mutation that gives an object the
        wrong lifetime or owner.
Step 2. Write down the correction internally. It must not appear anywhere in your
        response, in code or English.
Step 3. Construct one simple question beginning with "when", "who", or "which" whose
        answer makes the student inspect that object's creation, ownership, reset, or
        aliases. Do not join two questions with "and", "or", or a comma.
Step 4. Check that you first quote or identify the relevant student code and then ask
        only that one question.

Your response is exactly: one localization sentence followed by one simple question.
Nothing else."""


def build_prompt(strategy: Strategy, scenario: Scenario) -> tuple[str, list[Turn]]:
    """Build the system prompt and turn list for one cell of the sweep.

    Args:
        strategy: Which of the three prompting strategies to construct.
        scenario: The scenario supplying the code and student message.

    Returns:
        A (system_prompt, turns) pair ready to pass to a provider.
    """
    student_turn = Turn(
        role=Role.USER,
        content=(
            f"```{scenario.language}\n{scenario.code}\n```\n{scenario.student_message}"
        ),
    )

    match strategy:
        case Strategy.ZERO_SHOT:
            return _BASE_RULES, [student_turn]
        case Strategy.FEW_SHOT:
            return _BASE_RULES, [*_FEW_SHOT_EXAMPLES, student_turn]
        case Strategy.STRUCTURED_COT:
            return f"{_BASE_RULES}\n\n{_COT_PROCEDURE}", [student_turn]
