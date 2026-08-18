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
