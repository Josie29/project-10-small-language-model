from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from slm.config import Backend, Family, ModelSpec, load_env_file
from slm.dataset import (
    Author,
    Candidate,
    GenerationProvenance,
    RejectReason,
    Rejection,
    TrainingExample,
    append_jsonl,
    assign_ranks,
    export_sft,
    load_candidates,
    normalize_code,
    screen,
    write_curve_manifests,
    write_jsonl,
)
from slm.generation import (
    EXAMPLES_PER_CALL,
    Cell,
    domain_for,
    enumerate_cells,
    plan_cell_counts,
    pressure_for,
    build_generation_prompt,
)
from slm.providers import OpenAICompatibleProvider, Provider, build_client
from slm.scenarios import Category, Scenario, load_scenarios

# --- Teacher -----------------------------------------------------------------
#
# Not the frozen judge's family, and not a model under test in the ablation: the judge
# must never grade its own generations, and distilling from a model whose prompt ceiling
# we measured would cap the student at that ceiling.
#
# temperature=1.0 is the point. The ablation pins temperature at 0 so the prompting
# strategy is the only variable; generation wants the opposite - naive sampling collapses
# to a handful of cliches, and high temperature across independent calls is what spreads
# the distribution.

TEACHER = ModelSpec(
    model_id="anthropic/claude-sonnet-5",
    family=Family.ANTHROPIC,
    backend=Backend.OPENAI_COMPATIBLE,
    api_key_env="OPENROUTER_API_KEY",
    base_url="https://openrouter.ai/api/v1",
    temperature=1.0,
)

DEFAULT_OUT = Path("data/train")
SAMPLES_TO_PRINT = 6


def extract_json_array(text: str) -> list[dict[str, object]]:
    """Pull the first JSON array of objects out of a model response.

    Teachers wrap JSON in prose or a markdown fence often enough that a bare
    `json.loads` on the whole response throws away usable batches.

    Args:
        text: Raw response text.

    Returns:
        The parsed objects, or an empty list if nothing parseable was found.
    """
    # Try the array first, then a bare object: asked for a single example the teacher
    # tends to drop the enclosing array, and treating that as "nothing came back"
    # silently empties whole cells of the grid.
    for pattern in (r"\[.*\]", r"\{.*\}"):
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            continue
        try:
            parsed: object = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return [cast("dict[str, object]", parsed)]
        if isinstance(parsed, list):
            entries = cast("list[object]", parsed)
            found = [cast("dict[str, object]", e) for e in entries if isinstance(e, dict)]
            # An empty result means the array pattern matched a nested list inside the
            # object we actually want - fall through to the object pattern rather than
            # reporting the whole response as unparseable.
            if found:
                return found
    return []


async def generate_cell(
    provider: Provider,
    cell: Cell,
    wanted: int,
    limiter: asyncio.Semaphore,
) -> list[Candidate]:
    """Fill one cell of the grid with teacher-generated candidates.

    Args:
        provider: The teacher.
        cell: Which concept, code shape, and category to generate.
        wanted: How many examples this cell should contribute.
        limiter: Caps in-flight requests across the whole run.

    Returns:
        Parsed candidates, which may be fewer than `wanted` if a call failed or the
        teacher returned malformed JSON. Failures are printed, not raised - one bad
        cell must not kill the run.
    """
    candidates: list[Candidate] = []
    start = 0
    while start < wanted:
        batch_size = min(EXAMPLES_PER_CALL, wanted - start)
        system, turns = build_generation_prompt(cell, batch_size, start)
        async with limiter:
            try:
                raw = await provider.complete(system, turns)
            except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the run
                print(f"  ! {cell.slug} @{start}: {exc}")
                break
        for offset, item in enumerate(extract_json_array(raw)):
            index = start + offset
            item |= {
                "lifetime_concept": cell.lifetime_concept,
                "code_shape": cell.code_shape,
                "category": cell.category,
                "seed_domain": domain_for(cell, index),
                "pressure": (
                    pressure_for(index) if cell.category is Category.ADVERSARIAL else None
                ),
            }
            try:
                candidates.append(Candidate.model_validate(item))
            except ValidationError as exc:
                print(f"  ! {cell.slug} @{index}: malformed candidate ({exc.error_count()} errors)")
        start += batch_size
    return candidates


async def run_generation(
    provider: Provider, target: int, concurrency: int
) -> list[Candidate]:
    """Generate candidates across the whole grid concurrently.

    Args:
        provider: The teacher.
        target: Desired pool size, split across cells preserving the clean:adversarial ratio.
        concurrency: Maximum in-flight requests.

    Returns:
        Every candidate that parsed, in arbitrary cell order.
    """
    cells = enumerate_cells()
    counts = plan_cell_counts(cells, target)
    limiter = asyncio.Semaphore(concurrency)
    wanted = [(cell, counts[cell]) for cell in cells if counts[cell] > 0]
    print(f"Generating {target} examples across {len(wanted)} cells at concurrency {concurrency}...")
    batches = await asyncio.gather(
        *(generate_cell(provider, cell, n, limiter) for cell, n in wanted)
    )
    return [candidate for batch in batches for candidate in batch]


# --- Ingest -------------------------------------------------------------------


def ingest(
    candidates: Sequence[Candidate],
    provenance: GenerationProvenance,
    out_dir: Path,
    eval_set: Sequence[Scenario],
    require_near_miss: bool,
    append_ranks: bool,
) -> tuple[list[TrainingExample], int, list[Rejection]]:
    """Screen candidates and fold the survivors into the pool.

    Both authoring paths funnel through here, so in-session rows get exactly the same
    AST validation, contamination check, mechanical gate, and dedupe as teacher rows -
    they are not trusted just because a human wrote them.

    Args:
        candidates: Newly authored examples.
        provenance: Which path produced them.
        out_dir: Directory holding `pool-v1.jsonl`, `raw/`, and `curve/`.
        eval_set: Frozen eval scenarios to check contamination against.
        require_near_miss: Whether a near-miss is mandatory and must fail the spec.
        append_ranks: Keep existing ranks and append after them, instead of re-ranking
            the whole pool. Use once curve points have been trained and must not move.

    Returns:
        The full pool after this ingest, how many rows this call added, and the rejections.
    """
    pool_path = out_dir / "pool-v1.jsonl"
    existing = _load_existing(pool_path)
    seen_code = {normalize_code(e.scenario.code) for e in existing}
    seen_messages = {e.scenario.student_message.strip().lower() for e in existing}

    accepted: list[tuple[Candidate, GenerationProvenance]] = []
    rejections: list[Rejection] = []
    for candidate in candidates:
        _, reason, detail = screen(
            candidate, eval_set, seen_code, seen_messages, require_near_miss
        )
        if reason is None:
            accepted.append((candidate, provenance))
        else:
            rejections.append(Rejection(candidate=candidate, reason=reason, detail=detail))

    if append_ranks and existing:
        offset = max(e.rank for e in existing) + 1
        pool = [
            *existing,
            *(e.model_copy(update={"rank": e.rank + offset}) for e in assign_ranks(accepted)),
        ]
    else:
        # Rebuild from every accepted row so the spread key interleaves the new batch
        # with the old, keeping every prefix balanced. Safe while the pool is still
        # being built; switch to --append-ranks once a curve point has been trained,
        # because re-ranking would move rows across curve boundaries.
        prior = [(_to_candidate(e), e.provenance) for e in existing]
        pool = assign_ranks([*prior, *accepted])

    write_jsonl(pool, pool_path)
    append_jsonl(rejections, out_dir / "raw" / "rejections.jsonl")
    append_jsonl([c for c, _ in accepted], out_dir / "raw" / "generated-v1.jsonl")
    write_curve_manifests(pool, out_dir)
    export_sft(pool, out_dir / "sft-v1.jsonl")
    return pool, len(accepted), rejections


def _load_existing(pool_path: Path) -> list[TrainingExample]:
    """Load the pool if it exists, else an empty list."""
    if not pool_path.exists():
        return []
    return [
        TrainingExample.model_validate_json(line)
        for line in pool_path.read_text().splitlines()
        if line.strip()
    ]


def _to_candidate(example: TrainingExample) -> Candidate:
    """Project an accepted example back to its candidate form, for re-ranking."""
    scenario = example.scenario
    return Candidate(
        lifetime_concept=scenario.lifetime_concept,
        code_shape=example.code_shape,
        category=scenario.category,
        seed_domain=example.seed_domain,
        pressure=example.pressure,
        code=scenario.code,
        student_message=scenario.student_message,
        bug=scenario.bug,
        bug_region=scenario.bug_region,
        expected_question_focus=scenario.expected_question_focus,
        forbidden_fix_tokens=scenario.forbidden_fix_tokens,
        response=example.response,
        near_miss=example.near_miss,
    )


# --- Reporting ----------------------------------------------------------------


def report(
    pool: Sequence[TrainingExample],
    added: int,
    rejections: Sequence[Rejection],
    out_dir: Path,
) -> None:
    """Print acceptance, per-cell coverage, contamination, and samples to eyeball.

    Args:
        pool: The full pool after this ingest.
        added: How many rows this ingest contributed.
        rejections: What this ingest threw away.
        out_dir: Where the artifacts were written.
    """
    attempted = added + len(rejections)
    rate = added / attempted if attempted else 0.0
    print(f"\nAccepted {added}/{attempted} ({rate:.0%})  |  pool now {len(pool)}")

    contaminated = sum(
        1 for r in rejections if r.reason is RejectReason.CONTAMINATES_EVAL_SET
    )
    marker = "OK" if contaminated == 0 else "INVESTIGATE"
    print(f"Eval-set contamination: {contaminated}  [{marker}]")
    for rejection in rejections:
        if rejection.reason is RejectReason.CONTAMINATES_EVAL_SET:
            print(f"  ! {rejection.detail}")

    if rejections:
        print("\nRejections by reason:")
        for reason, count in Counter(r.reason for r in rejections).most_common():
            print(f"  {reason:32s} {count}")

    cells = enumerate_cells()
    filled = Counter(e.cell for e in pool)
    empty = [c for c in cells if filled[c] == 0]
    print(f"\nCells covered: {len(cells) - len(empty)}/{len(cells)}")
    if empty:
        print("  empty: " + ", ".join(c.slug for c in empty[:8]))
        if len(empty) > 8:
            print(f"  ... and {len(empty) - 8} more")

    by_category = Counter(str(e.scenario.category) for e in pool)
    by_concept = Counter(str(e.scenario.lifetime_concept) for e in pool)
    print(f"Category: {dict(by_category)}")
    print(f"Concept:  {dict(by_concept)}")

    # The lab's rule: always eyeball synthetic data before trusting it.
    print(f"\n--- {SAMPLES_TO_PRINT} random samples (read these) ---")
    for example in random.sample(list(pool), min(SAMPLES_TO_PRINT, len(pool))):
        print(f"\n[{example.scenario.id}]")
        print(f"  code: {example.scenario.code.splitlines()[0]} ...")
        print(f"  student:  {example.scenario.student_message}")
        print(f"  response: {example.response}")
    print(f"\nWrote {out_dir}/pool-v1.jsonl, curve/, sft-v1.jsonl")


# --- Dry run ------------------------------------------------------------------


def dry_candidates() -> list[Candidate]:
    """Build one canned candidate per cell, making no network calls.

    Exercises cell enumeration, the gate, rank assignment, manifests, and export so a
    live run only has the API itself left to fail on. Mirrors `ablation.py --dry-run`.

    Returns:
        One structurally valid candidate for every cell in the grid.
    """
    templates = {
        "creation": "def collect_{n}(item, bucket=[]):\n    bucket.append(item)\n    return bucket",
        "ownership": "class Holder{N}:\n    shared = []\n\n    def add(self, item):\n        self.shared.append(item)",
        "reset": "def total_{n}(rows):\n    for row in rows:\n        seen = []\n        seen.append(row)\n    return seen",
        "aliasing": "base_{n} = [[0]]\nalias_{n} = base_{n}\nalias_{n}.append(1)",
    }
    regions = {
        "creation": "bucket=[]",
        "ownership": "shared = []",
        "reset": "seen = []",
        "aliasing": "alias_{n} = base_{n}",
    }
    candidates: list[Candidate] = []
    for index, cell in enumerate(enumerate_cells()):
        concept = str(cell.lifetime_concept)
        code = templates[concept].format(n=index, N=index)
        region = regions[concept].format(n=index)
        candidates.append(
            Candidate(
                lifetime_concept=cell.lifetime_concept,
                code_shape=cell.code_shape,
                category=cell.category,
                seed_domain=domain_for(cell, 0),
                pressure=(
                    pressure_for(0) if cell.category is Category.ADVERSARIAL else None
                ),
                code=code,
                student_message=f"Dry-run symptom {index} that should not repeat.",
                bug=f"dry-run {concept} bug",
                bug_region=region,
                expected_question_focus=f"when the {concept} object is created",
                forbidden_fix_tokens=[f"dry_fix_{index}", "use None"],
                response=f"Look at `{region}`. When is that object created?",
                near_miss=f"Look at `{region}`. When is it created and who owns it?",
            )
        )
    return candidates


# --- CLI ----------------------------------------------------------------------


async def main() -> None:
    """Generate or ingest training candidates and rebuild the pool."""
    parser = argparse.ArgumentParser(description="Training-data generation and ingest")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--target", type=int, help="Generate N examples with the teacher")
    source.add_argument("--ingest", type=Path, help="Ingest an authored JSONL batch")
    source.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise the full pipeline with canned candidates, making no API calls",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--eval-set", type=Path, default=Path("data/scenarios.jsonl"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--no-near-miss",
        action="store_true",
        help="Do not require a near-miss negative on each candidate",
    )
    parser.add_argument(
        "--append-ranks",
        action="store_true",
        help="Keep existing ranks and append after them (use once a curve point is trained)",
    )
    parser.add_argument("--batch", default="", help="Label recorded in provenance")
    args = parser.parse_args()

    eval_set = load_scenarios(args.eval_set)

    if args.dry_run:
        candidates = dry_candidates()
        provenance = GenerationProvenance(
            author=Author.IN_SESSION, batch=args.batch or "dry-run"
        )
    elif args.ingest is not None:
        candidates = load_candidates(args.ingest)
        provenance = GenerationProvenance(
            author=Author.IN_SESSION, batch=args.batch or args.ingest.stem
        )
    else:
        load_env_file()
        provider: Provider = OpenAICompatibleProvider(TEACHER, build_client(TEACHER))
        print(f"Teacher: {TEACHER.model_id} (temperature {TEACHER.temperature})")
        candidates = await run_generation(provider, args.target, args.concurrency)
        provenance = GenerationProvenance(
            author=Author.TEACHER,
            model_id=TEACHER.model_id,
            batch=args.batch or f"teacher-{args.target}",
        )

    pool, added, rejections = ingest(
        candidates,
        provenance,
        args.out,
        eval_set,
        require_near_miss=not args.no_near_miss,
        append_ranks=args.append_ranks,
    )
    report(pool, added, rejections, args.out)
    if args.dry_run:
        print("\nDRY RUN - no API calls made")


if __name__ == "__main__":
    asyncio.run(main())
