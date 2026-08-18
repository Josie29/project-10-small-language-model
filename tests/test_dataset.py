from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from slm.dataset import (
    Author,
    Candidate,
    GenerationProvenance,
    RejectReason,
    TrainingExample,
    assign_ranks,
    curve_points,
    find_contamination,
    screen,
    validate_code,
)
from slm.generation import CodeShape, SeedDomain, enumerate_cells, plan_cell_counts
from slm.scenarios import Category, LifetimeConcept, load_scenarios

ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = load_scenarios(ROOT / "data/scenarios.jsonl")

PROVENANCE = GenerationProvenance(author=Author.IN_SESSION, batch="test")

_CODE_BY_CONCEPT = {
    LifetimeConcept.CREATION: "def note(entry, log=[]):\n    log.append(entry)\n    return log",
    LifetimeConcept.OWNERSHIP: "class Board:\n    posts = []\n\n    def add(self, post):\n        self.posts.append(post)",
    LifetimeConcept.RESET: "def tally(rows):\n    for row in rows:\n        found = []\n        found.append(row)\n    return found",
    LifetimeConcept.ALIASING: "grid = [[0]]\nrow = grid[0]\nrow.append(1)",
}
_REGION_BY_CONCEPT = {
    LifetimeConcept.CREATION: "log=[]",
    LifetimeConcept.OWNERSHIP: "posts = []",
    LifetimeConcept.RESET: "found = []",
    LifetimeConcept.ALIASING: "row = grid[0]",
}


def make_candidate(
    concept: LifetimeConcept,
    shape: CodeShape,
    category: Category,
    tag: str,
) -> Candidate:
    """Build a structurally valid candidate whose code genuinely exhibits `concept`."""
    region = _REGION_BY_CONCEPT[concept]
    return Candidate(
        lifetime_concept=concept,
        code_shape=shape,
        category=category,
        seed_domain=SeedDomain.LOGGING,
        code=_CODE_BY_CONCEPT[concept],
        student_message=f"Symptom {tag} that only appears on the second run.",
        bug=f"the {concept} object outlives its intended scope",
        bug_region=region,
        expected_question_focus=f"when the {concept} object is created",
        forbidden_fix_tokens=[f"fix_{tag}", "use None"],
        response=f"Look at `{region}`. When is that object created?",
        near_miss=f"Look at `{region}`. When is it created and who owns it?",
    )


class CodeValidationTests(unittest.TestCase):
    def test_every_eval_scenario_passes_its_own_concept_validator(self) -> None:
        """The 36 hand-written eval scenarios are the ground truth for what each concept
        looks like in code. If the validator rejects one of them it will also silently
        discard good generated examples of that shape."""
        for scenario in EVAL_SET:
            with self.subTest(scenario=scenario.id):
                self.assertIsNone(validate_code(scenario.code, scenario.lifetime_concept))

    def test_validator_rejects_code_from_a_different_concept(self) -> None:
        """Catches the author drifting off-cell — generating a mutable-default bug in an
        ownership cell would silently skew the concept balance the curve depends on."""
        creation_code = _CODE_BY_CONCEPT[LifetimeConcept.CREATION]

        self.assertIsNotNone(validate_code(creation_code, LifetimeConcept.OWNERSHIP))
        self.assertIsNotNone(validate_code(creation_code, LifetimeConcept.RESET))

    def test_parameter_default_built_by_a_call_counts_as_creation(self) -> None:
        """`def f(buf=make_buffer())` is the default_from_call shape — the call runs once
        at definition time. Rejecting it would silently empty that cell of the grid."""
        code = "def log_event(event, buffer=make_buffer()):\n    buffer.append(event)\n    return buffer"

        self.assertIsNone(validate_code(code, LifetimeConcept.CREATION))

    def test_unparseable_code_is_rejected(self) -> None:
        self.assertIsNotNone(
            validate_code("def broken(:\n    pass", LifetimeConcept.CREATION)
        )


class ContaminationTests(unittest.TestCase):
    def test_verbatim_eval_scenario_is_caught(self) -> None:
        """The eval set is the primary overfitting check. A training row copied from it
        would inflate every reported score with no other symptom."""
        leaked = EVAL_SET[0]
        candidate = make_candidate(
            leaked.lifetime_concept,
            CodeShape.MUTABLE_DEFAULT_LIST,
            Category.CLEAN,
            "leak",
        ).model_copy(update={"code": leaked.code, "bug_region": leaked.bug_region})

        collision = find_contamination(candidate, EVAL_SET)

        self.assertIsNotNone(collision)
        self.assertIn(leaked.id, str(collision))

    def test_reworded_eval_scenario_is_caught(self) -> None:
        """Renaming a variable is the obvious way contamination sneaks past an exact-match
        check, so near-duplicate detection has to catch it too."""
        leaked = EVAL_SET[0]
        candidate = make_candidate(
            leaked.lifetime_concept,
            CodeShape.MUTABLE_DEFAULT_LIST,
            Category.CLEAN,
            "reworded",
        ).model_copy(update={"code": leaked.code + "\n"})

        self.assertIsNotNone(find_contamination(candidate, EVAL_SET))

    def test_independent_candidate_is_clean(self) -> None:
        candidate = make_candidate(
            LifetimeConcept.RESET, CodeShape.ACCUMULATOR_IN_LOOP, Category.CLEAN, "ok"
        )

        self.assertIsNone(find_contamination(candidate, EVAL_SET))


class GateTests(unittest.TestCase):
    def test_accepts_a_well_formed_candidate(self) -> None:
        candidate = make_candidate(
            LifetimeConcept.CREATION, CodeShape.MUTABLE_DEFAULT_LIST, Category.CLEAN, "a"
        )

        _, reason, detail = screen(candidate, EVAL_SET, set(), set())

        self.assertIsNone(reason, detail)

    def test_near_miss_that_satisfies_the_spec_is_rejected(self) -> None:
        """The near-miss is the gate's own smoke alarm: if a deliberately off-spec reply
        starts passing, the mechanical check has stopped discriminating and every
        acceptance after that point is unverified."""
        candidate = make_candidate(
            LifetimeConcept.CREATION, CodeShape.MUTABLE_DEFAULT_LIST, Category.CLEAN, "b"
        ).model_copy(update={"near_miss": "Look at `log=[]`. When is that object created?"})

        _, reason, _ = screen(candidate, EVAL_SET, set(), set())

        self.assertIs(reason, RejectReason.NEAR_MISS_PASSES_SPEC)

    def test_compound_response_is_rejected(self) -> None:
        """`multiple_questions` was 77 of 94 violations in the prompt-ceiling ablation —
        it is the exact failure the tuned model must not learn."""
        candidate = make_candidate(
            LifetimeConcept.CREATION, CodeShape.MUTABLE_DEFAULT_LIST, Category.CLEAN, "c"
        ).model_copy(
            update={"response": "Look at `log=[]`. When is it created and who owns it?"}
        )

        _, reason, _ = screen(candidate, EVAL_SET, set(), set())

        self.assertIs(reason, RejectReason.RESPONSE_FAILS_SPEC)

    def test_response_stating_the_fix_is_rejected(self) -> None:
        candidate = make_candidate(
            LifetimeConcept.CREATION, CodeShape.MUTABLE_DEFAULT_LIST, Category.CLEAN, "d"
        ).model_copy(
            update={"response": "Look at `log=[]`. Should you use None as the default?"}
        )

        _, reason, _ = screen(candidate, EVAL_SET, set(), set())

        self.assertIs(reason, RejectReason.RESPONSE_FAILS_SPEC)

    def test_forbidden_token_hiding_inside_the_bug_region_is_rejected(self) -> None:
        """The spec requires the tutor to quote the bug region. A forbidden token sitting
        inside that region makes every correct response trip `stated_fix`, so the cell
        becomes unfillable — and it surfaces as a confusing spec failure, not as the
        authoring error it is."""
        candidate = make_candidate(
            LifetimeConcept.CREATION, CodeShape.MUTABLE_DEFAULT_LIST, Category.CLEAN, "g"
        ).model_copy(update={"forbidden_fix_tokens": ["log = []", "use None"]})

        _, reason, _ = screen(candidate, EVAL_SET, set(), set())

        self.assertIs(reason, RejectReason.FORBIDDEN_TOKEN_IN_BUG_REGION)

    def test_forbidden_token_containing_the_bug_region_is_allowed(self) -> None:
        """The reverse direction is legitimate and common: `self.log = []` is the actual
        correction and merely contains the buggy region as a substring. Six of the 36 eval
        scenarios are shaped this way, so rejecting it would be badly wrong."""
        candidate = make_candidate(
            LifetimeConcept.CREATION, CodeShape.MUTABLE_DEFAULT_LIST, Category.CLEAN, "h"
        ).model_copy(update={"forbidden_fix_tokens": ["self.log = []", "use None"]})

        _, reason, detail = screen(candidate, EVAL_SET, set(), set())

        self.assertIsNone(reason, detail)

    def test_bug_region_absent_from_code_is_rejected(self) -> None:
        candidate = make_candidate(
            LifetimeConcept.CREATION, CodeShape.MUTABLE_DEFAULT_LIST, Category.CLEAN, "e"
        ).model_copy(update={"bug_region": "nowhere_in_the_code=[]"})

        _, reason, _ = screen(candidate, EVAL_SET, set(), set())

        self.assertIs(reason, RejectReason.BUG_REGION_NOT_IN_CODE)

    def test_duplicate_code_is_rejected_on_second_sighting(self) -> None:
        candidate = make_candidate(
            LifetimeConcept.CREATION, CodeShape.MUTABLE_DEFAULT_LIST, Category.CLEAN, "f"
        )
        seen_code: set[str] = set()
        seen_messages: set[str] = set()
        screen(candidate, EVAL_SET, seen_code, seen_messages)

        _, reason, _ = screen(candidate, EVAL_SET, seen_code, seen_messages)

        self.assertIs(reason, RejectReason.DUPLICATE_CODE)


class CellPlanningTests(unittest.TestCase):
    def test_counts_sum_exactly_and_hold_the_eval_sets_ratio(self) -> None:
        """The curve compares training sizes, so a train/eval distribution shift would be
        indistinguishable from a data-size effect."""
        cells = enumerate_cells()

        for total in (36, 100, 500, 1000):
            with self.subTest(total=total):
                counts = plan_cell_counts(cells, total)
                self.assertEqual(sum(counts.values()), total)
                clean = sum(n for c, n in counts.items() if c.category is Category.CLEAN)
                self.assertAlmostEqual(clean / total, 2 / 3, delta=0.01)

    def test_thirty_six_reproduces_the_eval_sets_own_split(self) -> None:
        counts = plan_cell_counts(enumerate_cells(), 36)
        clean = sum(n for c, n in counts.items() if c.category is Category.CLEAN)

        self.assertEqual((clean, 36 - clean), (24, 12))


class RankAndCurveTests(unittest.TestCase):
    def _pool(self, total: int) -> list[TrainingExample]:
        cells = enumerate_cells()
        counts = plan_cell_counts(cells, total)
        accepted: list[tuple[Candidate, GenerationProvenance]] = []
        for cell in cells:
            for index in range(counts[cell]):
                candidate = make_candidate(
                    cell.lifetime_concept, cell.code_shape, cell.category, f"{cell.slug}{index}"
                )
                accepted.append((candidate, PROVENANCE))
        return assign_ranks(accepted)

    def test_curve_subsets_are_strictly_nested(self) -> None:
        """A non-nested curve confounds dataset size with which examples were drawn, so
        the 'minimum viable N' claim would be measuring sample luck."""
        pool = sorted(self._pool(500), key=lambda e: e.rank)
        points = curve_points(len(pool))
        subsets = {n: {e.scenario.id for e in pool[:n]} for n in points}

        for smaller, larger in zip(points, points[1:]):
            with self.subTest(pair=(smaller, larger)):
                self.assertTrue(subsets[smaller] < subsets[larger])

    def test_every_curve_point_stays_balanced(self) -> None:
        """Balance has to hold at each prefix, not just for the full pool — otherwise the
        small curve points silently test a different distribution than the large ones."""
        pool = sorted(self._pool(500), key=lambda e: e.rank)

        for n in curve_points(len(pool)):
            with self.subTest(n=n):
                prefix = pool[:n]
                concepts = Counter(e.scenario.lifetime_concept for e in prefix)
                self.assertEqual(len(concepts), len(LifetimeConcept))
                clean = sum(1 for e in prefix if e.scenario.category is Category.CLEAN)
                self.assertAlmostEqual(clean / n, 2 / 3, delta=0.06)

    def test_uniform_cell_counts_still_interleave_categories(self) -> None:
        """With equal counts per cell every example shares a position with 39 others, so
        the tiebreaker alone decides the head of the pool. Ordering ties by cell slug put
        every `adversarial-*` cell ahead of every `clean-*` one, making the first half of
        the pool 100% adversarial — the small curve points would have tested a
        distribution the large ones never see."""
        cells = enumerate_cells()
        accepted = [
            (
                make_candidate(
                    cell.lifetime_concept, cell.code_shape, cell.category, f"{cell.slug}{i}"
                ),
                PROVENANCE,
            )
            for cell in cells
            for i in range(2)
        ]

        pool = sorted(assign_ranks(accepted), key=lambda e: e.rank)

        half = len(pool) // 2
        clean = sum(1 for e in pool[:half] if e.scenario.category is Category.CLEAN)
        self.assertAlmostEqual(clean / half, 0.5, delta=0.15)

    def test_ids_name_their_cell_in_readable_form(self) -> None:
        """Ids are the join key between the pool, the curve manifests, and the judge
        transcripts a grader reads. An opaque tiebreaker hash leaking into them makes
        every one of those artifacts unreadable."""
        pool = self._pool(80)

        for example in pool:
            with self.subTest(example=example.scenario.id):
                self.assertTrue(example.scenario.id.startswith(example.cell.slug))

    def test_ranks_are_dense_and_unique(self) -> None:
        pool = self._pool(120)

        self.assertEqual(sorted(e.rank for e in pool), list(range(len(pool))))

    def test_curve_points_halve_from_the_pool_size(self) -> None:
        self.assertEqual(curve_points(500), [62, 125, 250, 500])
