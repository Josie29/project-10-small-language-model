from __future__ import annotations

import hashlib
import re
import unittest

from slm.judge import OPTIONAL_BLOCKS, build_judge_prompt
from slm.scenarios import Category, LifetimeConcept, Scenario

# SHA-256 of the fully-specified judge prompt as of commit 685085f, before any of the
# held-out-set work. Reproduce the pre-change value with:
#
#     git show 685085f:slm/judge.py
#
# The digest covers JUDGE_RUBRIC because the rubric is interpolated into the prompt, so
# this pins slm/spec.py as well. That is what judge.py's "must not be edited between the
# ablation and the eval without re-running both" comment is really protecting: every
# number in results/ was produced by this exact string, and nothing else guarded it.
PINNED_PROMPT_SHA256 = "3829c8bc7ef0ec511e9444db1b6ec38f5803fed496002d837e26b0dee10986c7"

FULL = Scenario(
    id="fixture-01",
    category=Category.CLEAN,
    language="python",
    code="C",
    student_message="M",
    bug="B",
    bug_region="R",
    lifetime_concept=LifetimeConcept.RESET,
    expected_question_focus="F",
    forbidden_fix_tokens=["t"],
)

SPARSE = Scenario(id="staff-01", code="C", student_message="M")


def _tags(prompt: str) -> set[str]:
    """Return the names of the XML-ish sections present in a judge prompt."""
    return set(re.findall(r"<([a-z_]+)[ >]", prompt))

class JudgePromptTests(unittest.TestCase):
    def test_fully_specified_prompt_is_unchanged(self) -> None:
        # The load-bearing test of the whole sparse-scenario change. Every score in
        # results/ was produced by this exact prompt; if assembling it conditionally
        # moves one character, the ablation and the eval are no longer comparable and
        # every published number silently becomes unreproducible.
        prompt = build_judge_prompt(FULL, "RESP")

        digest = hashlib.sha256(prompt.encode()).hexdigest()

        self.assertEqual(digest, PINNED_PROMPT_SHA256)

    def test_missing_fields_remove_their_blocks_and_nothing_else(self) -> None:
        # Guards against interpolating "None" into the judge prompt, which would tell
        # the judge the expected bug region is literally the string None - worse than
        # not asking about localization at all.
        full = build_judge_prompt(FULL, "RESP")
        sparse = build_judge_prompt(SPARSE, "RESP")

        for _, tag in OPTIONAL_BLOCKS:
            self.assertNotIn(tag, sparse)
        self.assertNotIn("None", sparse)
        self.assertIn("behavior spec alone", sparse)
        # Exactly the three optional sections leave; every other section stays. Compared
        # as tag sets rather than lengths because the degraded-mode note is longer than
        # the blocks it replaces.
        self.assertEqual(
            _tags(full) - _tags(sparse), {tag for _, tag in OPTIONAL_BLOCKS}
        )
        self.assertEqual(_tags(sparse) - _tags(full), set())

    def test_each_optional_block_is_individually_removable(self) -> None:
        # Catches a future rename of a prompt tag: _drop_block raises rather than
        # silently leaving the block in with a None substituted into it.
        for field, tag in OPTIONAL_BLOCKS:
            with self.subTest(field=field):
                partial = FULL.model_copy(update={field: None})

                prompt = build_judge_prompt(partial, "RESP")

                self.assertNotIn(f"<{tag}", prompt)
                self.assertNotIn("None", prompt)

    def test_only_bug_missing_keeps_the_rubric_blocks(self) -> None:
        # A partially-specified set is the interesting middle case: dropping `bug` must
        # not drag the expected-region blocks out with it, or a scenario that could have
        # been sharply graded gets scored in degraded mode for no reason.
        partial = FULL.model_copy(update={"bug": None})

        prompt = build_judge_prompt(partial, "RESP")

        self.assertNotIn("<actual_bug", prompt)
        self.assertIn("<expected_bug_region>", prompt)
        self.assertNotIn("behavior spec alone", prompt)


if __name__ == "__main__":
    unittest.main()
