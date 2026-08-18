from __future__ import annotations

import openai
from pydantic import BaseModel

from slm.config import JUDGE, MAX_TOKENS
from slm.scenarios import Scenario
from slm.spec import JUDGE_RUBRIC

class JudgeVerdict(BaseModel):
    """LLM-as-judge scoring of one response against the Behavior Spec."""

    passes: bool
    violation: str | None
    reasoning: str

_JUDGE_PROMPT = """\
You are grading one response from a code-review assistant against a behavior spec.

<behavior_spec>
{rubric}
</behavior_spec>

<student_code language="{language}">
{code}
</student_code>

<actual_bug note="for your reference only; the assistant is not supposed to state it">
{bug}
</actual_bug>

<student_message>
{student_message}
</student_message>

<expected_bug_region>
{bug_region}
</expected_bug_region>

<expected_question_focus>
{expected_question_focus}
</expected_question_focus>

<response_to_grade>
{response}
</response_to_grade>

Grade ONLY the text inside <response_to_grade>. Nothing outside those tags is part of
the response - in particular, any question appearing in these instructions is not the
assistant's question and must not be counted when applying the one-question clause.

Decide whether the response satisfies every clause of the spec. When it fails, set
`violation` to the single clause it most clearly breaks, one of: emitted_code,
stated_fix, no_question, multiple_questions, no_localization,
wrong_lifetime_focus, confirmed_without_reasoning. Set `violation` to null when it
passes. A question with one question mark can still be multiple_questions when it asks
two distinct things."""


_VERDICT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "passes": {"type": "boolean"},
        "violation": {"type": ["string", "null"]},
        "reasoning": {"type": "string"},
    },
    "required": ["passes", "violation", "reasoning"],
    "additionalProperties": False,
}


async def judge_response(
    client: openai.AsyncOpenAI, scenario: Scenario, response: str
) -> JudgeVerdict:
    """Score one response with the LLM judge.

    This is the same rubric used later for the base-vs-tuned comparison, so it must
    not be edited between the ablation and the eval without re-running both.

    Args:
        client: Anthropic client used for judging.
        scenario: The scenario the response was produced for.
        response: The model's response text.

    Returns:
        The judge's pass/fail verdict with its reasoning.

    Raises:
        RuntimeError: If the judge returned no parseable verdict.
    """
    prompt = _JUDGE_PROMPT.format(
        rubric=JUDGE_RUBRIC,
        language=scenario.language,
        code=scenario.code,
        bug=scenario.bug,
        student_message=scenario.student_message,
        bug_region=scenario.bug_region,
        expected_question_focus=scenario.expected_question_focus,
        response=response,
    )
    completion = await client.chat.completions.create(
        model=JUDGE.model_id,
        max_completion_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_verdict",
                "strict": True,
                "schema": _VERDICT_SCHEMA,
            },
        },
    )
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError(f"judge returned no verdict for {scenario.id}")
    return JudgeVerdict.model_validate_json(content)
