from __future__ import annotations

import unittest
from pathlib import Path

from slm.dataset import TRAIN_SYSTEM_PROMPT, load_pool
from slm.prompting import Strategy, build_prompt
from slm.training import TrainConfig, curve_subset, sft_rows

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data/train/pool-v1.jsonl"
CURVE_DIR = ROOT / "data/train/curve"


class CurveSubsetTests(unittest.TestCase):
    def test_curve_points_are_nested(self) -> None:
        # The whole claim of the data-efficiency curve is that dataset size is the only
        # variable between points. If a smaller point ever contains an example a larger one
        # does not, the curve is measuring which examples were drawn, not how many.
        pool = load_pool(POOL)
        sizes = sorted(int(p.stem.removeprefix("n-")) for p in CURVE_DIR.glob("n-*.txt"))

        previous: set[str] = set()
        for size in sizes:
            ids = {e.scenario.id for e in curve_subset(pool, size)}

            self.assertEqual(len(ids), size)
            self.assertTrue(previous <= ids, f"n-{size} is not a superset of the point below")
            previous = ids

    def test_curve_subset_matches_the_committed_manifest(self) -> None:
        # generate.py writes the manifests and train.py re-derives the same prefix from
        # ranks. If those two ever disagree, the published dataset stops describing the
        # checkpoints that were actually trained.
        pool = load_pool(POOL)
        for manifest in sorted(CURVE_DIR.glob("n-*.txt")):
            size = int(manifest.stem.removeprefix("n-"))
            expected = manifest.read_text().split()

            derived = [e.scenario.id for e in curve_subset(pool, size)]

            self.assertEqual(derived, expected, f"{manifest.name} drifted from rank order")

    def test_asking_for_more_than_the_pool_holds_raises(self) -> None:
        pool = load_pool(POOL)

        with self.assertRaises(ValueError):
            curve_subset(pool, len(pool) + 1)


class SftExportTests(unittest.TestCase):
    def test_user_turn_is_identical_to_what_the_base_model_is_prompted_with(self) -> None:
        # This is the experiment's control. The base model is evaluated with the full
        # behavior-spec system prompt and the tuned model with one throwaway line; if the
        # *user* turns also differed, a base-vs-tuned delta would have a second possible
        # explanation and the comparison would prove nothing.
        pool = load_pool(POOL)
        for example in curve_subset(pool, 20):
            row = sft_rows([example])[0]
            messages = row["messages"]
            _, turns = build_prompt(Strategy.ZERO_SHOT, example.scenario)

            self.assertEqual(messages[1]["content"], turns[-1].content)
            self.assertEqual(messages[0]["content"], TRAIN_SYSTEM_PROMPT)

    def test_rows_come_out_in_rank_order(self) -> None:
        pool = load_pool(POOL)
        subset = curve_subset(pool, 30)

        rows = sft_rows(list(reversed(subset)))

        self.assertEqual(
            [r["messages"][2]["content"] for r in rows],
            [e.response for e in subset],
        )


class TrainConfigTests(unittest.TestCase):
    def test_step_count_reflects_the_curve_point(self) -> None:
        # The small end of the curve trains for very few optimizer steps, which is a
        # confound worth surfacing in the report rather than discovering afterwards.
        small = TrainConfig(base_model="b", dataset_size=62, repo_id="r")
        large = TrainConfig(base_model="b", dataset_size=500, repo_id="r")

        self.assertEqual(small.expected_steps, (62 // 8) * 3)
        self.assertEqual(large.expected_steps, (500 // 8) * 3)
        self.assertGreaterEqual(small.warmup_steps, 1)
        self.assertLess(small.warmup_steps, small.expected_steps)
