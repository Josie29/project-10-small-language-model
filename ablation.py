from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

import anthropic
import openai

from slm.checks import run_mechanical_check
from slm.config import (
    DEFAULT_CONCURRENCY,
    JUDGE,
    Backend,
    Family,
    ModelSpec,
    load_env_file,
)
from slm.judge import JudgeVerdict, judge_response
from slm.prompting import Strategy, build_prompt
from slm.providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    Provider,
    build_client,
)
from slm.reporting import Trial, write_results
from slm.scenarios import Scenario, load_scenarios, stratified_sample

# --- Models under test -------------------------------------------------------
#
# Two models from different families. Staff cleared a small model for slot A. Note the
# ablation's claim is only as strong as the models tested: a ceiling shown on cheap
# models is weaker evidence that *prompting* is the limit rather than model capacity.

MODELS_UNDER_TEST: list[ModelSpec] = [
    ModelSpec(
        model_id="anthropic/claude-haiku-4.5",
        family=Family.ANTHROPIC,
        backend=Backend.OPENAI_COMPATIBLE,
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
    ),
    ModelSpec(
        # k2.5 is cheaper but is withdrawn 2026-08-31, and a grader re-running this in
        # September would get a 404. Pinned to Moonshot's own serve: OpenRouter spreads
        # this model across ~21 providers running int4 through bf16, so an unpinned run
        # would not reproduce.
        model_id="moonshotai/kimi-k2.6",
        family=Family.MOONSHOT,
        backend=Backend.OPENAI_COMPATIBLE,
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        extra_body={
            "provider": {"order": ["moonshotai/int4"], "allow_fallbacks": False},
            "reasoning": {"enabled": False},
        },
    ),
]

# --- Sweep -------------------------------------------------------------------


async def run_trial(
    provider: Provider,
    strategy: Strategy,
    scenario: Scenario,
    judge_client: openai.AsyncOpenAI,
    limiter: asyncio.Semaphore,
) -> Trial | None:
    """Run one (model, strategy, scenario) cell and score it.

    Args:
        provider: The model under test.
        strategy: The prompting strategy for this cell.
        scenario: The scenario to run.
        judge_client: Client for the judge call.
        limiter: Caps in-flight requests across the whole sweep.

    Returns:
        The scored trial, or None if the call failed (the failure is printed and the
        scenario is dropped from that cell's denominator).
    """
    system, turns = build_prompt(strategy, scenario)
    async with limiter:
        try:
            response = await provider.complete(system, turns)
            verdict = await judge_response(judge_client, scenario, response)
        except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the sweep
            print(f"  ! {provider.model_id}/{strategy}/{scenario.id}: {exc}")
            return None
    return Trial(
        scenario_id=scenario.id,
        category=scenario.category,
        lifetime_concept=scenario.lifetime_concept,
        model_id=provider.model_id,
        family=provider.family,
        strategy=strategy,
        response=response,
        check=run_mechanical_check(response, scenario),
        verdict=verdict,
    )


async def run_sweep(
    providers: Sequence[Provider],
    scenarios: Sequence[Scenario],
    judge_client: openai.AsyncOpenAI,
    concurrency: int,
) -> list[Trial]:
    """Run every model x strategy x scenario combination concurrently.

    Args:
        providers: Models under test — at least two, from different families.
        scenarios: The scenario set.
        judge_client: Client for judge calls.
        concurrency: Maximum in-flight requests.

    Returns:
        Every trial that completed, in arbitrary order.
    """
    limiter = asyncio.Semaphore(concurrency)
    tasks = [
        run_trial(provider, strategy, scenario, judge_client, limiter)
        for provider in providers
        for strategy in Strategy
        for scenario in scenarios
    ]
    print(f"Running {len(tasks)} trials at concurrency {concurrency}...")
    results = await asyncio.gather(*tasks)
    return [trial for trial in results if trial is not None]

# --- Dry run -----------------------------------------------------------------

def dry_trial(spec_id: str, family: Family, strategy: Strategy, scenario: Scenario) -> Trial:
    """Build one trial from a canned response, making no network calls.

    Exercises prompt construction, the deterministic check, aggregation and table
    rendering so a live run only has the API itself left to fail on.

    Args:
        spec_id: Model id to record on the trial.
        family: Family to record on the trial.
        strategy: Strategy for this cell.
        scenario: Scenario being run.

    Returns:
        A fully-populated trial whose verdict is derived from the mechanical check.
    """
    build_prompt(strategy, scenario)  # exercise prompt construction
    response = (
        f"Look at `{scenario.bug_region}`. "
        "When does that object begin its current lifetime?"
    )
    check = run_mechanical_check(response, scenario)
    return Trial(
        scenario_id=scenario.id,
        category=scenario.category,
        lifetime_concept=scenario.lifetime_concept,
        model_id=spec_id,
        family=family,
        strategy=strategy,
        response=response,
        check=check,
        verdict=JudgeVerdict(
            passes=check.passed,
            violation=None if check.passed else "emitted_code",
            reasoning="dry run - no judge call made",
        ),
    )


async def main() -> None:
    """Run the ablation and write trials plus the results table to disk."""
    parser = argparse.ArgumentParser(description="Prompt-ceiling ablation")
    parser.add_argument("--scenarios", type=Path, default=Path("data/scenarios.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/state-lifetime-v1"))
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run the first N scenarios (smoke test)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise the full pipeline with canned responses, making no API calls",
    )
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenarios)
    if args.limit is not None:
        scenarios = stratified_sample(scenarios, args.limit)

    if args.dry_run:
        trials = [
            dry_trial(spec.model_id, spec.family, strategy, scenario)
            for spec in MODELS_UNDER_TEST
            for strategy in Strategy
            for scenario in scenarios
        ]
        write_results(trials, args.out)
        print(f"DRY RUN - {len(trials)} trials, no API calls made")
        return

    load_env_file()
    providers: list[Provider] = []
    for spec in MODELS_UNDER_TEST:
        if spec.backend is Backend.ANTHROPIC:
            providers.append(
                AnthropicProvider(spec, anthropic.AsyncAnthropic())
            )
        else:
            providers.append(OpenAICompatibleProvider(spec, build_client(spec)))

    judge_client = build_client(JUDGE)
    print(
        f"Models: {', '.join(p.model_id for p in providers)}  |  Judge: {JUDGE.model_id}"
    )

    trials = await run_sweep(providers, scenarios, judge_client, args.concurrency)

    write_results(trials, args.out)


if __name__ == "__main__":
    asyncio.run(main())
