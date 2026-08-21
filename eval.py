from __future__ import annotations

import argparse
import asyncio
import subprocess
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

import openai
from pydantic import BaseModel, Field

from slm.checkpoints import DEFAULT_CHECKPOINTS, infer_dataset_size, load_checkpoints
from slm.checks import canned_response, run_mechanical_check
from slm.config import (
    BASE_MODEL,
    DEFAULT_CONCURRENCY,
    JUDGE,
    load_env_file,
)
from slm.dataset import TRAIN_SYSTEM_PROMPT
from slm.judge import JudgeVerdict, judge_response
from slm.local import TransformersProvider, local_spec, resolve_device
from slm.prompting import Strategy, build_prompt
from slm.providers import Provider, Turn, build_client
from slm.reporting import Trial, write_results
from slm.scenarios import (
    RubricCoverage,
    Scenario,
    load_scenarios_with_coverage,
    rubric_coverage,
    stratified_sample,
)

DEFAULT_OUT = Path("results/base-vs-tuned")
RESULTS_TITLE = "Base vs Tuned — Results"

# --- Targets -----------------------------------------------------------------


class TargetRole(StrEnum):
    """Which side of the comparison a model sits on.

    The roles differ only in their system prompt, and that difference is the experiment.
    The base model is given the strongest prompt the ablation found; the tuned model is
    given the one throwaway line it was trained against. A tuned model that needs no
    prompt engineering to beat a prompted base is the claim being tested.
    """

    BASE = "base"
    TUNED = "tuned"


class EvalTarget(BaseModel):
    """One model in the base-vs-tuned comparison."""

    model_id: str
    role: TargetRole
    dataset_size: int | None = None
    # Hub commit sha. Pinning it is what makes a reported number reproducible rather than
    # merely repeatable against whatever `main` happens to be later.
    revision: str | None = None

    @property
    def system_prompt_label(self) -> str:
        """Which prompt this target is evaluated with, for the run manifest."""
        return "behavior_spec_zero_shot" if self.role is TargetRole.BASE else "train_system_prompt"


class ScoringMode(StrEnum):
    """How much of the answer key was available when the numbers were produced."""

    SCENARIO_RUBRIC = "scenario_rubric"
    SPEC_ONLY = "spec_only"


class EvalRun(BaseModel):
    """Everything a grader needs to reproduce one results table.

    Written alongside the trials because the brief's "pinned versions" requirement is not
    satisfied by a score table: the numbers only mean something against a named eval set,
    a named judge, and specific model and code commits.
    """

    eval_set: str
    n_scenarios: int
    judge_model: str
    eval_code_commit: str
    device: str
    targets: list[EvalTarget]
    # All three default so a manifest written before this existed still parses, and so
    # a fully-specified run's manifest reads exactly as it always did.
    scoring_mode: ScoringMode = ScoringMode.SCENARIO_RUBRIC
    rubric_coverage: RubricCoverage | None = None
    # Per model, scenarios whose generation or judge call failed and were dropped.
    trials_dropped: dict[str, int] = Field(default_factory=dict)


def degraded_banner(coverage: RubricCoverage) -> str:
    """Build the warning printed before a run scored without a full answer key.

    Printed before any weights load so an operator learns the run is degraded in the
    first second rather than after forty minutes of generation.

    Args:
        coverage: Per-field answer-key counts for the eval set.

    Returns:
        The banner text, or an empty string when the set is fully specified.
    """
    if not coverage.degraded:
        return ""
    n = coverage.n_scenarios
    missing = n - min(coverage.with_bug_region, coverage.with_expected_question_focus)
    lines = [
        "",
        "=" * 78,
        f"DEGRADED SCORING - {missing} of {n} scenarios carry no bug_region or",
        "expected_question_focus.",
        "",
        "The judge grades those against the behavior spec alone: it sees the code and",
        "the response but not which line is the bug, so it decides wrong_lifetime_focus",
        "and no_localization from its own reading. The mechanical clauses those fields",
        "feed drop out of the mechanical pass rate.",
    ]
    if coverage.defaulted_to_clean:
        lines += [
            "",
            f"{coverage.defaulted_to_clean} scenarios carry no category and are counted "
            "as clean. Spec adherence is",
            "measured over them; Robustness is measured only over scenarios explicitly",
            "labelled adversarial. Neither number is a claim about which they are.",
        ]
    lines += [
        "",
        "Every omission removes evidence a response could have FAILED on, never",
        "evidence it could have passed on. These numbers are biased high and are not",
        "comparable to anything in results/.",
        "=" * 78,
        "",
    ]
    return "\n".join(lines)


def build_eval_prompt(role: TargetRole, scenario: Scenario) -> tuple[str, list[Turn]]:
    """Build the system prompt and turn list for one target.

    Both roles are handed the identical user turn, produced by the same `build_prompt`
    call the ablation used - so the only difference between base and tuned input is the
    system prompt, and nothing else can quietly explain a delta.

    Args:
        role: Which side of the comparison this model is on.
        scenario: The scenario supplying the code and student message.

    Returns:
        A (system_prompt, turns) pair ready to pass to a provider.
    """
    system, turns = build_prompt(Strategy.ZERO_SHOT, scenario)
    if role is TargetRole.TUNED:
        system = TRAIN_SYSTEM_PROMPT
    return system, turns


def resolve_targets(
    model_ids: Sequence[str], base: str | None, checkpoints_path: Path
) -> list[EvalTarget]:
    """Turn CLI model arguments into targets, enriched from the checkpoint manifest.

    Args:
        model_ids: Tuned checkpoints to evaluate, in the order given.
        base: The base model to compare against, or None to skip it.
        checkpoints_path: Manifest written by `train.py`.

    Returns:
        The base target first, if requested, then one target per tuned checkpoint.

    Raises:
        ValueError: If no targets were requested at all.
    """
    manifest = {c.repo_id: c for c in load_checkpoints(checkpoints_path)}
    targets: list[EvalTarget] = []
    if base is not None:
        targets.append(EvalTarget(model_id=base, role=TargetRole.BASE))
    for model_id in model_ids:
        recorded = manifest.get(model_id)
        targets.append(
            EvalTarget(
                model_id=model_id,
                role=TargetRole.TUNED,
                dataset_size=(
                    recorded.dataset_size if recorded else infer_dataset_size(model_id)
                ),
                revision=recorded.revision if recorded else None,
            )
        )
    if not targets:
        raise ValueError("nothing to evaluate: pass --model, or drop --no-base")
    return targets


def eval_code_commit() -> str:
    """Return the git commit this eval ran from, marked dirty if the tree has changes.

    `results/` is excluded from the dirtiness check. A rerun writes its own output into
    that tracked directory before stamping this field, so counting it would mark every
    single rerun dirty no matter how clean the checkout was - the run reporting on itself.
    Changes anywhere else, including the eval set and `slm/`, still mark the run dirty.

    Returns:
        The short sha, suffixed `-dirty` when uncommitted changes are present outside
        `results/`, or "unknown" outside a git checkout.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", ":(exclude)results"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


# --- Sweep -------------------------------------------------------------------


async def run_trial(
    provider: Provider,
    target: EvalTarget,
    scenario: Scenario,
    judge_client: openai.AsyncOpenAI,
    limiter: asyncio.Semaphore,
) -> Trial | None:
    """Run one scenario against one target and score it.

    Args:
        provider: The loaded model.
        target: Which model this is and what role it plays.
        scenario: The scenario to run.
        judge_client: Client for the judge call.
        limiter: Caps in-flight judge requests.

    Returns:
        The scored trial, or None if the call failed (the failure is printed and the
        scenario is dropped from that model's denominator).
    """
    system, turns = build_eval_prompt(target.role, scenario)
    # Generation and judging are caught separately so the printed line names which half
    # failed. A systematic judge failure - wrong key, wrong base URL - drops every trial
    # and yields an empty table; diagnosing that should not require reading source.
    try:
        response = await provider.complete(system, turns)
    except Exception as exc:  # noqa: BLE001 - one bad scenario must not kill the sweep
        print(f"  ! {target.model_id}/{scenario.id}: generation failed: {exc}")
        return None
    try:
        async with limiter:
            verdict = await judge_response(judge_client, scenario, response)
    except Exception as exc:  # noqa: BLE001 - one bad scenario must not kill the sweep
        print(f"  ! {target.model_id}/{scenario.id}: judge failed: {exc}")
        return None
    return Trial(
        scenario_id=scenario.id,
        category=scenario.category,
        lifetime_concept=scenario.lifetime_concept,
        model_id=target.model_id,
        family=provider.family,
        # Both roles are prompted zero-shot; the ablation's other two strategies are not
        # part of this comparison. Recorded rather than omitted because Trial requires it,
        # and it keeps these rows renderable by the same table code.
        strategy=Strategy.ZERO_SHOT,
        response=response,
        check=run_mechanical_check(response, scenario),
        verdict=verdict,
        dataset_size=target.dataset_size,
    )


async def run_sweep(
    targets: Sequence[EvalTarget],
    scenarios: Sequence[Scenario],
    judge_client: openai.AsyncOpenAI,
    concurrency: int,
    device: str,
) -> list[Trial]:
    """Evaluate every target on every scenario.

    Targets run one at a time so only one set of weights is resident: five 0.6B
    checkpoints held at once would not fit comfortably alongside a laptop's other work,
    and generation against a single model does not parallelise anyway. Judge calls for a
    finished scenario overlap the next generation.

    Args:
        targets: Models to evaluate, base first.
        scenarios: The eval set.
        judge_client: Client for judge calls.
        concurrency: Maximum in-flight judge requests.
        device: Torch device for the in-process models.

    Returns:
        Every trial that completed, in target order.
    """
    limiter = asyncio.Semaphore(concurrency)
    trials: list[Trial] = []
    for target in targets:
        print(f"\n{target.role}: {target.model_id} ({len(scenarios)} scenarios)")
        provider = TransformersProvider(
            local_spec(target.model_id), device=device, revision=target.revision
        )
        results = await asyncio.gather(
            *(
                run_trial(provider, target, scenario, judge_client, limiter)
                for scenario in scenarios
            )
        )
        scored = [trial for trial in results if trial is not None]
        if len(scored) < len(scenarios):
            # Loud, because the alternative is a plausible rate over a silently smaller
            # denominator. The count reaches run.json and the results table too.
            print(
                f"  ! {target.model_id}: scored {len(scored)} of {len(scenarios)} "
                f"scenarios - {len(scenarios) - len(scored)} dropped"
            )
        trials += scored
        del provider
        release_device_memory(device)
    return trials


def release_device_memory(device: str) -> None:
    """Drop the previous target's weights before loading the next one.

    Dereferencing alone leaves the allocator holding the blocks, so a five-model sweep
    would accumulate every checkpoint it had already finished with.

    Args:
        device: The torch device in use.
    """
    import gc

    import torch

    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


async def preflight_judge(
    judge_client: openai.AsyncOpenAI, scenario: Scenario
) -> None:
    """Make one judge call before any weights load, to fail fast on a bad client.

    `run_trial` swallows per-scenario failures so one bad scenario cannot kill an hour of
    generation - which means a misconfigured judge drops every trial and produces an
    empty table instead of an error. One call up front turns that into one sentence.

    Args:
        judge_client: Client for judge calls.
        scenario: Any scenario; only used to build a well-formed request.

    Raises:
        RuntimeError: If the judge could not be reached or returned nothing usable.
    """
    try:
        await judge_response(judge_client, scenario, canned_response(scenario))
    except Exception as exc:
        raise RuntimeError(
            f"judge preflight failed against {JUDGE.model_id}: {exc}\n"
            f"Every trial would be dropped and the results table would come out empty. "
            f"Check the API key and base URL in .env before rerunning."
        ) from exc
    print(f"Judge preflight OK ({JUDGE.model_id})")


# --- Dry run -----------------------------------------------------------------


def dry_trial(target: EvalTarget, scenario: Scenario) -> Trial:
    """Build one trial from a canned response, making no network or model calls.

    Exercises prompt construction, the deterministic check, aggregation, and both table
    renderers so a live run only has model loading and the judge API left to fail on.

    Args:
        target: The model this trial stands in for.
        scenario: Scenario being run.

    Returns:
        A fully-populated trial whose verdict is derived from the mechanical check.
    """
    build_eval_prompt(target.role, scenario)  # exercise prompt construction
    response = canned_response(scenario)
    check = run_mechanical_check(response, scenario)
    return Trial(
        scenario_id=scenario.id,
        category=scenario.category,
        lifetime_concept=scenario.lifetime_concept,
        model_id=target.model_id,
        family=local_spec(target.model_id).family,
        strategy=Strategy.ZERO_SHOT,
        response=response,
        check=check,
        verdict=JudgeVerdict(
            passes=check.passed,
            violation=None if check.passed else "emitted_code",
            reasoning="dry run - no judge call made",
        ),
        dataset_size=target.dataset_size,
    )


async def main() -> None:
    """Evaluate the base model and one or more tuned checkpoints on the same eval set."""
    parser = argparse.ArgumentParser(description="Base-vs-tuned behavior evaluation")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="HF_REPO_ID",
        help="A tuned checkpoint to evaluate. Repeat for a data-efficiency curve.",
    )
    parser.add_argument(
        "--base",
        default=BASE_MODEL,
        help="Untuned model to compare against, prompted with the full behavior spec",
    )
    parser.add_argument(
        "--no-base",
        action="store_true",
        help="Skip the base model (it is slow, and its numbers do not change)",
    )
    parser.add_argument("--eval-set", type=Path, default=Path("data/scenarios.jsonl"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument(
        "--device", default=None, help="torch device; autodetected when omitted"
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run the first N scenarios (smoke test)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise the full pipeline with canned responses, loading no weights",
    )
    args = parser.parse_args()

    scenarios, coverage = load_scenarios_with_coverage(args.eval_set)
    if args.limit is not None:
        scenarios = stratified_sample(scenarios, args.limit)
        # Coverage describes whatever is actually being scored, not the file on disk.
        # The defaulted-to-clean count cannot survive sampling, so it is recomputed as 0
        # and the banner simply stops claiming it.
        coverage = rubric_coverage(scenarios)

    targets = resolve_targets(
        args.model, None if args.no_base else args.base, args.checkpoints
    )

    print(degraded_banner(coverage), end="")

    if args.dry_run:
        trials = [dry_trial(t, s) for t in targets for s in scenarios]
        write_results(
            trials,
            args.out,
            RESULTS_TITLE,
            coverage=coverage,
            n_scenarios=len(scenarios),
        )
        print(f"DRY RUN - {len(trials)} trials, no weights loaded and no API calls made")
        return

    load_env_file()
    device = resolve_device(args.device)
    judge_client = build_client(JUDGE)
    print(
        f"Eval set: {args.eval_set} ({len(scenarios)} scenarios)  |  "
        f"Judge: {JUDGE.model_id}  |  Device: {device}"
    )

    await preflight_judge(judge_client, scenarios[0])

    trials = await run_sweep(targets, scenarios, judge_client, args.concurrency, device)

    write_results(
        trials, args.out, RESULTS_TITLE, coverage=coverage, n_scenarios=len(scenarios)
    )
    dropped = {
        t.model_id: len(scenarios)
        - sum(1 for trial in trials if trial.model_id == t.model_id)
        for t in targets
    }
    run = EvalRun(
        eval_set=str(args.eval_set),
        n_scenarios=len(scenarios),
        judge_model=JUDGE.model_id,
        eval_code_commit=eval_code_commit(),
        device=device,
        targets=list(targets),
        scoring_mode=(
            ScoringMode.SCENARIO_RUBRIC
            if not coverage.degraded
            else ScoringMode.SPEC_ONLY
        ),
        rubric_coverage=coverage,
        trials_dropped={k: v for k, v in dropped.items() if v},
    )
    run_path = args.out / "run.json"
    run_path.write_text(run.model_dump_json(indent=2) + "\n")
    print(f"Run manifest -> {run_path}")


if __name__ == "__main__":
    asyncio.run(main())
