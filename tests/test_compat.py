from __future__ import annotations

import unittest
from pathlib import Path

from slm.dataset import TrainingExample
from slm.reporting import Trial, aggregate, render_table
from slm.scenarios import load_scenarios

ROOT = Path(__file__).resolve().parents[1]

# Title passed to render_table when each result set was written. The prompt-ceiling
# ablation and the base-vs-tuned eval share a renderer but not a heading.
_TITLES = {
    "state-lifetime-v1": "Prompt-Ceiling Ablation — Results",
}
_DEFAULT_TITLE = "Base vs Tuned — Results"

class CommittedArtifactTests(unittest.TestCase):
    def test_every_committed_trial_still_parses(self) -> None:
        # Scenario's answer-key fields became optional and two MechanicalCheck clauses
        # became tri-state so a held-out eval set can be loaded. If that widening broke
        # deserialization, every raw judge transcript the brief requires us to submit
        # would be unreadable by the code that wrote it.
        files = sorted((ROOT / "results").glob("*/trials.jsonl"))
        self.assertTrue(files, "no committed trials.jsonl found")

        for path in files:
            with self.subTest(path=path.name, dir=path.parent.name):
                lines = [l for l in path.read_text().splitlines() if l.strip()]

                trials = [Trial.model_validate_json(line) for line in lines]

                self.assertEqual(len(trials), len(lines))

    def test_committed_tables_still_render_identically(self) -> None:
        # The published numbers, locked. Any change to the schema, the aggregation, or
        # the renderer that would move a figure in a committed table.md fails here
        # instead of shipping — which is the whole reason the results directories are
        # tracked rather than gitignored.
        for table_path in sorted((ROOT / "results").glob("*/table.md")):
            with self.subTest(dir=table_path.parent.name):
                directory = table_path.parent
                trials = [
                    Trial.model_validate_json(line)
                    for line in (directory / "trials.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                title = _TITLES.get(directory.name, _DEFAULT_TITLE)

                rendered = render_table(aggregate(trials), trials, title)

                self.assertEqual(rendered, table_path.read_text())

    def test_every_committed_scenario_set_still_loads(self) -> None:
        # The eval set, the shape-swap probe, and the contamination-check union are all
        # inputs a grader reruns. A loader change that rejects one of them breaks the
        # reproduce-from-nothing claim the brief's verification section rests on.
        for name in (
            "data/scenarios.jsonl",
            "data/probe-shape-swap.jsonl",
            "data/heldout-union.jsonl",
        ):
            with self.subTest(name=name):
                scenarios = load_scenarios(ROOT / name)

                self.assertTrue(scenarios)

    def test_committed_training_pool_still_parses(self) -> None:
        # TrainingExample embeds Scenario, so widening Scenario reaches the published
        # dataset too — the artifact the brief calls "your real artifact".
        for pool in sorted((ROOT / "data").glob("train*/pool-v1.jsonl")):
            with self.subTest(pool=pool.parent.name):
                lines = [l for l in pool.read_text().splitlines() if l.strip()]

                examples = [TrainingExample.model_validate_json(l) for l in lines]

                self.assertEqual(len(examples), len(lines))


if __name__ == "__main__":
    unittest.main()
