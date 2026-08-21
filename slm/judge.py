from __future__ import annotations

import re

import openai
from pydantic import BaseModel, ValidationError

from slm.config import JUDGE, MAX_TOKENS
from slm.scenarios import Scenario
from slm.spec import JUDGE_RUBRIC

# The judge occasionally degenerates into emitting blank lines, runs into
# `max_completion_tokens`, and returns JSON cut off mid-object. Measured on one probe
# scenario it happens on roughly one call in three, which silently cost a trial in two
# separate runs before it was tracked down.
#
# Retrying is safe for comparability: a truncated reply carries no verdict at all, so
# there is nothing to select between. This re-asks for a verdict, it does not re-ask for
# a better one - the retry condition is "unparseable", never "the wrong answer".
JUDGE_ATTEMPTS = 3

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


# Which scenario field supplies each removable block of the prompt. A held-out set will
# not carry an answer key, and interpolating None into the template would tell the judge
# the expected region is literally "None" - worse than not asking about it at all.
OPTIONAL_BLOCKS: tuple[tuple[str, str], ...] = (
    ("bug", "actual_bug"),
    ("bug_region", "expected_bug_region"),
    ("expected_question_focus", "expected_question_focus"),
)

# Degraded mode only. Without an expected region the judge is deciding localization from
# its own reading of the code, and it must not report that as if it had an answer key.
_NO_RUBRIC_NOTE = (
    "\n\nThis scenario supplies no reference bug region or expected question focus. "
    "Judge the response against the behavior spec alone. Identify the bug yourself "
    "from the student code, and only use `no_localization` or `wrong_lifetime_focus` "
    "when the response is clearly wrong on your own reading - not merely unconfirmed."
)


def _drop_block(template: str, tag: str) -> str:
    """Remove one `<tag>...</tag>` section and its trailing blank line.

    Args:
        template: The judge prompt template.
        tag: Name of the XML-ish tag to remove.

    Returns:
        The template without that section.

    Raises:
        RuntimeError: If the tag was not found. A template edit that renames a tag would
            otherwise silently stop degrading and start rendering "None" at the judge.
    """
    # `[^>]*` because <actual_bug> carries a note= attribute.
    pattern = re.compile(rf"<{tag}[^>]*>\n.*?\n</{tag}>\n\n", re.DOTALL)
    stripped, count = pattern.subn("", template, count=1)
    if count != 1:
        raise RuntimeError(
            f"judge prompt has no <{tag}> block to remove - the template and "
            f"OPTIONAL_BLOCKS have drifted apart"
        )
    return stripped


def build_judge_prompt(scenario: Scenario, response: str) -> str:
    """Assemble the judge prompt, omitting blocks the scenario cannot fill.

    A fully-specified scenario takes no branch: the frozen template is formatted exactly
    as it always was, so the prompt is byte-identical to the one every number in
    `results/` was produced with. Blocks are only ever *removed*, never rebuilt.

    Args:
        scenario: The scenario the response was produced for.
        response: The model's response text.

    Returns:
        The prompt to send to the judge.
    """
    template = _JUDGE_PROMPT
    for field, tag in OPTIONAL_BLOCKS:
        if getattr(scenario, field) is None:
            template = _drop_block(template, tag)
    if not scenario.has_rubric:
        template += _NO_RUBRIC_NOTE
    return template.format(
        rubric=JUDGE_RUBRIC,
        language=scenario.language,
        code=scenario.code,
        # Removed placeholders are simply absent from the template; str.format ignores
        # kwargs it has no field for, so no per-field bookkeeping is needed here.
        bug=scenario.bug or "",
        student_message=scenario.student_message,
        bug_region=scenario.bug_region or "",
        expected_question_focus=scenario.expected_question_focus or "",
        response=response,
    )


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
    prompt = build_judge_prompt(scenario, response)
    last_failure = ""
    for _ in range(JUDGE_ATTEMPTS):
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
        choice = completion.choices[0]
        content = choice.message.content
        if not content:
            last_failure = "empty response"
            continue
        if choice.finish_reason == "length":
            # Truncated mid-object. Parsing would fail anyway; say so precisely rather
            # than reporting it as malformed JSON.
            last_failure = f"hit the {MAX_TOKENS}-token cap and was truncated"
            continue
        try:
            return JudgeVerdict.model_validate_json(content)
        except ValidationError as exc:
            last_failure = f"unparseable verdict: {exc}"
    raise RuntimeError(
        f"judge returned no usable verdict for {scenario.id} after "
        f"{JUDGE_ATTEMPTS} attempts - {last_failure}"
    )
