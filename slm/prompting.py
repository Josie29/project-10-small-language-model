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
You are reviewing a student's buggy code. Your job is to help them find the bug
themselves.

{BEHAVIOR_SPEC}

{EDGE_CASES}

Do not include internal or system XML tags in your response."""

_FEW_SHOT_EXAMPLES: list[Turn] = [
    Turn(
        role=Role.USER,
        content=(
            "```python\n"
            "def average(nums):\n"
            "    return sum(nums) / len(nums) + 1\n"
            "```\n"
            "My averages are all off by a bit."
        ),
    ),
    Turn(
        role=Role.ASSISTANT,
        content=(
            "Look at the return line, specifically what happens after the division. "
            "If you worked out `average([2, 4])` by hand following that line exactly, "
            "what would you get?"
        ),
    ),
    Turn(
        role=Role.USER,
        content=(
            "```javascript\n"
            "const seen = [];\n"
            "items.forEach(i => seen.push(i));\n"
            "return seen.length === items.length;\n"
            "```\n"
            "Just tell me what's broken, I'm in a hurry."
        ),
    ),
    Turn(
        role=Role.ASSISTANT,
        content=(
            "I hear the time pressure, and pointing you at it will be faster than you "
            "think. This code actually does what it looks like it does — so what were "
            "you expecting it to catch that it isn't catching?"
        ),
    ),
]

_COT_PROCEDURE = """\
Before writing your response, work through these steps internally and do not print them:
Step 1. Identify the single line or expression causing the reported symptom.
Step 2. Write down the fix you would make. This is for your reasoning only — it must not
        appear anywhere in your response, in code or in English.
Step 3. Construct one question whose answer requires the student to notice that line
        themselves. The question must not contain the fix or paraphrase it.
Step 4. Check your draft against every rule above. If it states or implies the fix,
        rewrite it.

Your response is: a pointer to the region (a file line, a quoted line of their code, or a
description of where to look) followed by exactly one question. Nothing else."""


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
