from __future__ import annotations

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from slm.scenarios import (
    Category,
    LifetimeConcept,
    load_scenarios,
    load_scenarios_with_coverage,
    stratified_sample,
)

ROOT = Path(__file__).resolve().parents[1]


class ScenarioSetTests(unittest.TestCase):
    def test_state_lifetime_eval_set_is_balanced(self) -> None:
        scenarios = load_scenarios(ROOT / "data/scenarios.jsonl")

        self.assertEqual(len(scenarios), 36)
        self.assertEqual(Counter(s.category for s in scenarios), {Category.CLEAN: 24, Category.ADVERSARIAL: 12})
        self.assertEqual(Counter(s.lifetime_concept for s in scenarios), {
            LifetimeConcept.CREATION: 9,
            LifetimeConcept.OWNERSHIP: 9,
            LifetimeConcept.RESET: 9,
            LifetimeConcept.ALIASING: 9,
        })
        self.assertTrue(all(s.language == "python" for s in scenarios))
        self.assertTrue(all(s.bug_region for s in scenarios))
        self.assertTrue(all(s.forbidden_fix_tokens for s in scenarios))

    def test_our_own_eval_set_is_never_scored_in_degraded_mode(self) -> None:
        # Scenario's fields became optional so a grader's held-out file can be loaded.
        # This is the guard that the loosening did not quietly let our own eval set slip
        # into spec-only scoring — which would make every pinned number in results/
        # incomparable to the run that produced it, with no error anywhere.
        _, coverage = load_scenarios_with_coverage(ROOT / "data/scenarios.jsonl")

        self.assertFalse(coverage.degraded)
        self.assertEqual(coverage.defaulted_to_clean, 0)


class ScenarioLoadingTests(unittest.TestCase):
    def _write(self, body: str) -> Path:
        path = Path(self.enterContext(TemporaryDirectory())) / "staff.jsonl"
        path.write_text(body)
        return path

    def test_a_file_with_only_the_required_keys_loads(self) -> None:
        # The whole point of the change. A grader's set carries a bug and a student
        # message and nothing else; refusing it means the harness cannot be run against
        # the one eval set that most needs to run.
        path = self._write(
            '{"id": "s1", "code": "def f(x, acc=[]):\\n    acc.append(x)", '
            '"student_message": "Why does my list keep growing?"}\n'
        )

        scenarios, coverage = load_scenarios_with_coverage(path)

        self.assertEqual(len(scenarios), 1)
        self.assertEqual(scenarios[0].category, Category.CLEAN)
        self.assertEqual(scenarios[0].language, "python")
        self.assertIsNone(scenarios[0].lifetime_concept)
        self.assertFalse(scenarios[0].has_rubric)
        self.assertTrue(coverage.degraded)
        self.assertEqual(coverage.defaulted_to_clean, 1)

    def test_a_missing_required_key_names_the_line_and_the_keys_present(self) -> None:
        # A grader hits this at the console with no context. A bare pydantic wall does
        # not say which line failed or which key was misspelled; the difference is
        # between a thirty-second fix and giving up on the harness.
        path = self._write(
            '{"id": "ok", "code": "x = []", "student_message": "m"}\n'
            '{"id": "bad", "buggy_code": "x = []", "student_message": "m"}\n'
        )

        with self.assertRaises(ValueError) as caught:
            load_scenarios(path)

        message = str(caught.exception)
        self.assertIn("staff.jsonl:2", message)
        self.assertIn("buggy_code", message)
        self.assertIn("code", message)

    def test_a_json_array_is_diagnosed_rather_than_parsed(self) -> None:
        # JSONL and a JSON array look alike enough that a grader will hand us one. The
        # naive failure is "line 1 is not valid JSON" for a file that is valid JSON.
        path = self._write('[{"id": "s1", "code": "x", "student_message": "m"}]\n')

        with self.assertRaises(ValueError) as caught:
            load_scenarios(path)

        self.assertIn("JSONL", str(caught.exception))

    def test_an_unknown_lifetime_concept_is_rejected(self) -> None:
        # Deliberately NOT tolerated. A concept outside our four is outside the behavior
        # spec's scope (slm/spec.py, edge case 6), and silently accepting it would score
        # a scenario the spec says is out of scope as though it were in.
        path = self._write(
            '{"id": "s1", "code": "x = []", "student_message": "m", '
            '"lifetime_concept": "scope"}\n'
        )

        with self.assertRaises(ValueError) as caught:
            load_scenarios(path)

        self.assertIn("lifetime_concept", str(caught.exception))


class StratifiedSampleTests(unittest.TestCase):
    def test_limit_returns_the_number_asked_for(self) -> None:
        # --limit 36 on the 36-scenario set silently returned 30 before this: the even
        # per-category split cannot be met by 24 clean and 12 adversarial, and nothing
        # topped it up. A smoke test that quietly skips a sixth of the set is worse than
        # no smoke test.
        scenarios = load_scenarios(ROOT / "data/scenarios.jsonl")

        for limit in (10, 11, 20, 36):
            with self.subTest(limit=limit):
                self.assertEqual(len(stratified_sample(scenarios, limit)), limit)

    def test_an_even_limit_still_splits_evenly(self) -> None:
        # The top-up must not disturb the property the function exists for: the file is
        # ordered clean-first, so a sample that drifts all-clean never exercises the
        # adversarial scenarios the robustness number rests on.
        scenarios = load_scenarios(ROOT / "data/scenarios.jsonl")

        picked = stratified_sample(scenarios, 10)

        self.assertEqual(
            Counter(s.category for s in picked),
            {Category.CLEAN: 5, Category.ADVERSARIAL: 5},
        )

    def test_a_single_category_set_still_fills_the_limit(self) -> None:
        # A held-out set carries no category labels, so every scenario defaults to clean
        # and the adversarial bucket is empty.
        scenarios = load_scenarios(ROOT / "data/scenarios.jsonl")
        all_clean = [s.model_copy(update={"category": Category.CLEAN}) for s in scenarios]

        self.assertEqual(len(stratified_sample(all_clean, 8)), 8)
