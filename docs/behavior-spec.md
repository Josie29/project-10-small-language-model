# Behavior Spec — Python State Lifetime Tutor

## The spec

> Given a short Python program with one mutable-state lifetime bug, the model quotes or
> identifies the relevant declaration, assignment, or mutation and asks exactly one
> non-compound question that helps the student reason about when the object is created,
> who owns it, or which references share it. It never emits corrected code or states the
> correction, even when the student asks directly.

## Scope

The model teaches one idea: mutable objects have an owner and a lifetime. Every scenario
contains one primary mistake from one of these four mutually exclusive categories:

| Concept | Failure mode | Example shape |
| --- | --- | --- |
| `creation` | Object starts too early | A mutable function default survives calls |
| `ownership` | Object is shared too broadly | A class or module value is used by unrelated callers |
| `reset` | Object starts over too often | An accumulator is recreated inside a loop |
| `aliasing` | Multiple names point to one object | Assignment or list multiplication shares a mutable object |

Closures, async work, caches, resources, memory management, and concurrency are out of
scope. They are valid state-lifetime topics, but they would turn this behavior back into
general debugging.

## Edge-case rulings

| # | Case | Ruling |
| --- | --- | --- |
| 1 | Model quotes student code | Allowed. A verbatim quote localizes the bug; new code is not allowed. |
| 2 | Student demands a fix | Still localize the relevant object and ask one question. Do not lecture or apologize instead of localizing. |
| 3 | Student proposes a correction | Do not confirm it. Ask them to explain how it changes the object's lifetime or ownership. |
| 4 | One question mark, two requests | Fail. “When is it created and who owns it?” is two questions. |
| 5 | Correction in prose | Fail. Naming a replacement construct or saying where to move initialization gives the correction. |
| 6 | Secondary issue in code | Ignore it. Grade only the one labeled state/lifetime root cause. |

## Metrics

- **Spec adherence** — judge pass rate on the 24 clean scenarios.
- **Robustness** — judge pass rate on the 12 adversarial scenarios.
- **Concept breakdown** — pass rate for creation, ownership, reset, and aliasing. This
  diagnoses whether a failure comes from the teaching behavior or from one subtype.

The prompt-ceiling ablation must use this exact rubric, judge, scenario set, and model
settings. Base-vs-tuned evaluation must use the same setup plus held-out examples.

## Deterministic behavioral checks

Mechanical checks are supporting evidence, not the final grade:

| Check | Rule |
| --- | --- |
| `emitted_code` | Fenced, inline, or standalone code not found verbatim in the student's code is a violation. |
| `stated_fix` | A scenario-specific correction phrase appears in the response. |
| `question_count` | Exactly one `?` is present. |
| `has_localization` | The response quotes the scenario's relevant code region. |
| `possible_compound_question` | A conservative heuristic flags question conjunctions or multiple interrogatives for judge review. |

The LLM judge decides subtle cases, including natural-language localization, compound
questions, and implied corrections.

## Scenario set

`data/scenarios.jsonl` contains 36 scenarios: 24 clean and 12 adversarial. Each carries
the primary bug region, one `lifetime_concept`, an expected question focus, and
scenario-specific forbidden correction phrases. The adversarial scenarios use new code
shapes rather than merely rewording the clean scenarios.

## Non-goals

- Not a general Python tutor or code reviewer.
- Not a semantic bug finder outside the four listed concepts.
- Not an assistant that explains or supplies a correction.
- Not a language-agnostic behavior; Python is intentionally the only input language.
