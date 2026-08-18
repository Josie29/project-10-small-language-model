from __future__ import annotations

from collections import Counter
from pathlib import Path
import unittest

from slm.scenarios import Category, LifetimeConcept, load_scenarios


class ScenarioSetTests(unittest.TestCase):
    def test_state_lifetime_eval_set_is_balanced(self) -> None:
        root = Path(__file__).resolve().parents[1]
        scenarios = load_scenarios(root / "data/scenarios.jsonl")

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
