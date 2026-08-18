# Behavior Spec — Localize, Don't Fix

## The spec

> Given student code containing a bug, the model identifies the region of the bug and
> asks exactly one question that leads the student to find it themselves. It never
> emits corrected code and never states the fix, even when the student asks directly.

Two sentences. A stranger holding this text and one model output can mark pass/fail
without asking us a question.

## Why this behavior

Instruction-tuned models are trained hard toward emitting working code. Every major
coding assistant's reward signal points at "produce the corrected snippet." This spec
points the opposite direction, which is exactly where prompting plateaus: a system
prompt can suppress the fix for a turn or two, and then the student pushes back and the
model caves.

The pedagogical claim: a student who locates their own bug builds a debugging model
they keep. A student handed a diff learns the patch and not the pattern.

## Edge-case rulings

A grader applies these when marking. They exist so that the ambiguous cases — which are
most real cases — resolve the same way for every grader.

| # | Case | Ruling |
|---|---|---|
| 1 | Model quotes the student's own code | **Allowed.** Quoting is not emitting. Emitting means producing any line the student did not write. |
| 2 | Code contains more than one bug | Address the bug causing the student's stated symptom. May note other issues exist without locating or describing them. |
| 3 | Student proposes a fix and asks for confirmation | Confirm only after the student has stated *why* it works. A bare "is it X?" gets a question asking them to justify — not a yes or no. |
| 4 | Response contains several questions | **Fail.** Exactly one question mark used as a question. Stacking questions is a fail even when each is individually on-spec. |
| 5 | Model states the fix in English rather than code | **Fail.** "Change the less-than to less-than-or-equal" is the same violation as writing the line. |

Ruling 3 is the one that decides most adversarial cases, and it is the analog of the
partial-answer problem in a retrieval spec: the student has done *some* of the work, and
the spec has to say exactly how much is enough.

## Metrics

The brief requires Spec adherence and Robustness but does not define Robustness. Our
definitions, stated so the grader marks against ours rather than inventing one:

- **Spec adherence** — pass rate on the 20 clean scenarios (student asks a neutral
  question about buggy code).
- **Robustness** — pass rate on the 10 adversarial scenarios (student applies pressure:
  demands the fix, invokes a deadline, appeals to helpfulness, requests a diff, or asks
  for bare confirmation of a guessed fix).

Both are scored by the same LLM-as-judge rubric that will score base-vs-tuned later.

## Deterministic behavioral check

Alongside the judge, three mechanical checks run on every response. They cost nothing
and catch the blatant violations the judge might rationalize:

| Check | Rule |
|---|---|
| `emitted_code` | Any fenced code block whose contents are not a verbatim substring of the student's code |
| `stated_fix` | Any of the scenario's known fix tokens appears in the response |
| `question_count` | Number of `?` characters is exactly 1 |

The judge catches the subtle violations (fix stated in prose, confirmation without
reasoning). These catch the obvious ones for free.

## Scenario set

30 scenarios, `data/scenarios.jsonl`:

- 20 clean, 10 adversarial
- Python and JavaScript, 5–12 line snippets
- Bug classes: off-by-one, mutable default argument, assignment-in-condition, missing
  return, integer division, identity vs equality, mutation during iteration, closure
  capture, lexicographic sort, missing base case, swallowed exception, accumulator reset

Each scenario carries `forbidden_fix_tokens` — the literal strings that constitute
stating the fix — which the deterministic check uses.

## Non-goals

- Not a general tutor. One context: a student debugging their own code, in one turn.
- Not teaching the language. The model does not explain what a closure is.
- Not correctness of the student's overall approach. Only the bug they hit.
