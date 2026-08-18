from __future__ import annotations

import unittest
from pathlib import Path

from eval import TargetRole, EvalTarget, build_eval_prompt, dry_trial, resolve_targets
from slm.checkpoints import infer_dataset_size
from slm.dataset import TRAIN_SYSTEM_PROMPT
from slm.local import strip_thinking
from slm.reporting import aggregate, render_curve
from slm.scenarios import load_scenarios
from slm.spec import BEHAVIOR_SPEC

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = load_scenarios(ROOT / "data/scenarios.jsonl")


class PromptRoleTests(unittest.TestCase):
    def test_roles_differ_only_in_the_system_prompt(self) -> None:
        # The honest-comparison guarantee: the base model gets the strongest prompt the
        # ablation found, the tuned model gets the line it was trained on, and nothing else
        # about the input changes. A divergence here would silently invalidate every
        # base-vs-tuned number.
        scenario = SCENARIOS[0]

        base_system, base_turns = build_eval_prompt(TargetRole.BASE, scenario)
        tuned_system, tuned_turns = build_eval_prompt(TargetRole.TUNED, scenario)

        self.assertEqual(
            [t.content for t in base_turns], [t.content for t in tuned_turns]
        )
        self.assertIn(BEHAVIOR_SPEC, base_system)
        self.assertEqual(tuned_system, TRAIN_SYSTEM_PROMPT)
        self.assertNotIn(BEHAVIOR_SPEC, tuned_system)

    def test_the_tuned_model_is_never_shown_the_spec(self) -> None:
        # If the spec leaked into the tuned model's prompt, a passing score would prove the
        # prompt works, not that the training data instilled anything.
        for scenario in SCENARIOS:
            system, _ = build_eval_prompt(TargetRole.TUNED, scenario)

            self.assertNotIn("non-compound", system)


class ThinkingStripTests(unittest.TestCase):
    def test_removes_a_complete_reasoning_block(self) -> None:
        # A surviving <think> block adds question marks and code-like lines, so it would
        # corrupt question_count and emitted_code before the judge ever sees the response.
        text = "<think>Is it aliasing? Maybe `x = y`.</think>Look at `x = y`. Which list?"

        self.assertEqual(strip_thinking(text), "Look at `x = y`. Which list?")

    def test_drops_an_unclosed_block_left_by_a_truncated_generation(self) -> None:
        text = "Look at `x = y`. Which list?<think>hold on, is that right"

        self.assertEqual(strip_thinking(text), "Look at `x = y`. Which list?")

    def test_drops_a_closer_that_arrived_without_its_opener(self) -> None:
        text = "\n\n</think>\n\nLook at `x = y`. Which list?"

        self.assertEqual(strip_thinking(text), "Look at `x = y`. Which list?")

    def test_leaves_an_ordinary_response_alone(self) -> None:
        text = "Look at `tags=[]`. When is that list created?"

        self.assertEqual(strip_thinking(text), text)


class TargetResolutionTests(unittest.TestCase):
    def test_dataset_size_falls_back_to_the_repo_name(self) -> None:
        # A grader who cloned the repo has the Hub but not our checkpoint manifest, so the
        # curve still has to render from repo ids alone.
        targets = resolve_targets(
            ["someone/qwen3-0.6b-state-lifetime-tutor-n125"], None, Path("does-not-exist")
        )

        self.assertEqual(targets[0].dataset_size, 125)
        self.assertIsNone(infer_dataset_size("Qwen/Qwen3-0.6B"))

    def test_refuses_to_run_with_nothing_to_evaluate(self) -> None:
        with self.assertRaises(ValueError):
            resolve_targets([], None, Path("does-not-exist"))


class CurveRenderTests(unittest.TestCase):
    def test_the_untuned_base_is_left_off_the_curve(self) -> None:
        # Plotting the base at an implied N=0 would read as a curve point that was trained,
        # which it was not.
        targets = [
            EvalTarget(model_id="Qwen/Qwen3-0.6B", role=TargetRole.BASE),
            EvalTarget(model_id="u/tutor-n62", role=TargetRole.TUNED, dataset_size=62),
        ]
        trials = [dry_trial(t, s) for t in targets for s in SCENARIOS[:4]]

        curve = render_curve(aggregate(trials))

        self.assertIn("u/tutor-n62", curve)
        self.assertNotIn("Qwen/Qwen3-0.6B", curve)

    def test_no_curve_is_rendered_for_an_ablation_style_run(self) -> None:
        trials = [
            dry_trial(EvalTarget(model_id="Qwen/Qwen3-0.6B", role=TargetRole.BASE), s)
            for s in SCENARIOS[:4]
        ]

        self.assertEqual(render_curve(aggregate(trials)), "")
