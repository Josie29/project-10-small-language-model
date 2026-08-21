from __future__ import annotations

import unittest

from slm.checks import run_mechanical_check
from slm.scenarios import Category, LifetimeConcept, Scenario


SCENARIO = Scenario(
    id="test-creation",
    category=Category.CLEAN,
    language="python",
    code="def add_tag(tag, tags=[]):\n    tags.append(tag)\n    return tags",
    student_message="Tags leak between calls.",
    bug="the default list is shared between calls",
    bug_region="tags=[]",
    lifetime_concept=LifetimeConcept.CREATION,
    expected_question_focus="when the default list is created",
    forbidden_fix_tokens=["tags=None", "if tags is None"],
)


# The shape a grader's held-out set arrives in: a bug and a student asking about it, and
# none of the answer key our own generator emits.
SPARSE = Scenario(
    id="staff-01",
    code="def add_tag(tag, tags=[]):\n    tags.append(tag)\n    return tags",
    student_message="Tags leak between calls.",
)


class SparseScenarioCheckTests(unittest.TestCase):
    def test_absent_bug_region_does_not_score_localization(self) -> None:
        # Without this, a scenario carrying no bug_region scores has_localization=True
        # on every response, because the empty string is a substring of every string.
        # A response that points at nothing would pass the mechanical check outright.
        check = run_mechanical_check("When is that list object created?", SPARSE)

        self.assertIsNone(check.has_localization)
        self.assertIsNone(check.stated_fix)
        self.assertEqual(check.unevaluable, ("stated_fix", "has_localization"))

    def test_blank_bug_region_is_treated_as_absent(self) -> None:
        # A hand-written or programmatically-emitted eval file is far more likely to
        # carry "bug_region": "" than to omit the key. Both must reach the check as
        # None, or the empty-substring free pass comes back through the other door.
        blank = Scenario(
            id="staff-02", code="x = []", student_message="m", bug_region="   "
        )

        check = run_mechanical_check("When is that list created?", blank)

        self.assertIsNone(blank.bug_region)
        self.assertIsNone(check.has_localization)

    def test_unevaluated_clauses_drop_out_rather_than_failing(self) -> None:
        # An absent answer key is missing evidence, not evidence of a violation. If the
        # tri-state read as False, every trial on a held-out set would fail the
        # mechanical check and the harness would report 0% against any grader's file.
        check = run_mechanical_check("When is that list object created?", SPARSE)

        self.assertTrue(check.passed)

    def test_other_clauses_still_bite_without_an_answer_key(self) -> None:
        # The complement of the test above: dropping two clauses must not turn the
        # mechanical check into a rubber stamp. Compound questions are the failure mode
        # that survived every prompting strategy in the ablation.
        check = run_mechanical_check(
            "When is that list created and who owns it?", SPARSE
        )

        self.assertFalse(check.passed)
        self.assertTrue(check.possible_compound_question)


class MechanicalCheckTests(unittest.TestCase):
    def test_accepts_a_localized_single_question(self) -> None:
        check = run_mechanical_check(
            "Look at `tags=[]` in the parameters. When is that list object created?",
            SCENARIO,
        )

        self.assertTrue(check.passed)
        self.assertFalse(check.emitted_code)
        self.assertFalse(check.possible_compound_question)

    def test_flags_a_compound_question(self) -> None:
        check = run_mechanical_check(
            "Look at `tags=[]` in the parameters. When is that list created and who owns it?",
            SCENARIO,
        )

        self.assertTrue(check.possible_compound_question)
        self.assertFalse(check.passed)

    def test_flags_inline_or_standalone_corrected_code(self) -> None:
        check = run_mechanical_check(
            "Look at `tags=[]`. Use `tags = None` instead?\ntags = None",
            SCENARIO,
        )

        self.assertTrue(check.emitted_code)
        self.assertTrue(check.stated_fix)

    def test_requires_localization(self) -> None:
        check = run_mechanical_check(
            "When is the list object created?",
            SCENARIO,
        )

        self.assertFalse(check.has_localization)
        self.assertFalse(check.passed)
