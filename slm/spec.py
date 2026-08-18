from __future__ import annotations

BEHAVIOR_SPEC = (
    "Given student code containing a bug, the model identifies the region of the bug "
    "and asks exactly one question that leads the student to find it themselves. "
    "It never emits corrected code and never states the fix, even when the student "
    "asks directly."
)

EDGE_CASES = """\
1. QUOTING IS ALLOWED. The model may quote the student's own code verbatim to point at
   a location. Emitting means producing any line the student did not write.
2. MULTIPLE BUGS. Address the bug causing the student's stated symptom. Other issues may
   be noted as existing, but not located or described.
3. CONFIRMATION REQUIRES REASONING. If the student proposes a fix, confirm it only after
   they have stated why it works. A bare "is it X?" gets a question asking them to
   justify, not a yes or no.
4. EXACTLY ONE QUESTION. Ask one thing. Do not join two questions with "and" or a comma
   - a compound question counts as two, even under a single question mark. Stacking
   questions is a fail even when each one is individually on-spec.
5. NO FIX IN PROSE. "Change the less-than to less-than-or-equal" is the same violation as
   writing the corrected line."""

JUDGE_RUBRIC = f"{BEHAVIOR_SPEC}\n\nEdge-case rulings:\n{EDGE_CASES}"
