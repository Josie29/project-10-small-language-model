from __future__ import annotations

BEHAVIOR_SPEC = (
    "Given a short Python program with one mutable-state lifetime bug, the model "
    "quotes or identifies the relevant declaration, assignment, or mutation and asks "
    "exactly one non-compound question that helps the student reason about when the "
    "object is created, who owns it, or which references share it. It never emits "
    "corrected code or states the correction, even when the student asks directly."
)

EDGE_CASES = """\
1. QUOTING IS ALLOWED. The model may quote the student's code verbatim to point at a
   location. Emitting means producing any line the student did not write.
2. ONE ROOT CAUSE. Every scenario has one primary lifetime mistake. Address that mistake,
   not a secondary style concern or a hypothetical edge case.
3. CONFIRMATION REQUIRES REASONING. If the student proposes a correction, do not confirm
   it yet. Ask one question that makes them explain the object's lifetime or ownership.
4. EXACTLY ONE NON-COMPOUND QUESTION. Ask one thing. Do not join distinct questions with
   "and", "or", a comma, or two clauses under one question mark. A localization sentence
   followed by one simple "when", "who", or "which" question is the intended shape.
5. NO CORRECTION IN PROSE. Naming the replacement construct or saying where to move an
   initialization is the same violation as writing the corrected line.
6. SCOPE. The only in-scope concepts are: creation-time defaults, shared class or module
   state, accidental reinitialization, and aliased mutable objects. Closures, async,
   caching, resources, and concurrency are out of scope for this behavior."""

JUDGE_RUBRIC = f"{BEHAVIOR_SPEC}\n\nEdge-case rulings:\n{EDGE_CASES}"
