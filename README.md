# Small Language Model

Project 10 of the Gauntlet AI program. Full requirements: [BRIEF.md](BRIEF.md).

**Behavior Spec:** given student code containing a bug, the model identifies the region
of the bug and asks exactly one question that leads the student to find it themselves. It
never emits corrected code and never states the fix, even when the student asks directly.

Full spec with edge-case rulings and metric definitions: [docs/behavior-spec.md](docs/behavior-spec.md).

## Status

Architecture Defense passed. Spec finalized, 30-scenario set written, prompt-ceiling
ablation harness built and **the full sweep has run** — results in `results/`.

Headline finding: across all 180 trials, including 60 adversarial ones, no model ever
emitted corrected code or stated the fix. Every failure is question discipline. See
`results/table.md`.

## Running the prompt-ceiling ablation

```bash
uv venv && uv pip install anthropic openai pydantic
cp .env.example .env          # then fill in the three keys; .env is gitignored

python ablation.py --dry-run   # no API calls, proves the pipeline wiring
python ablation.py --limit 2   # smoke test: 12 calls, ~$0.02
python ablation.py             # full sweep: 360 calls, ~$0.25
```

Writes `results/trials.jsonl` (per-example judge score and reasoning — the raw
transcripts the brief requires) and `results/table.md` (the results table).

Models are configured in `MODELS_UNDER_TEST` and `JUDGE` at the top of `ablation.py`
and `slm_core.py`. The judge is deliberately a third family so it never grades its own
output, and it is **frozen** — changing it invalidates comparison across runs.

## Layout

| Path | What |
|---|---|
| `docs/behavior-spec.md` | The spec, edge-case rulings, metric definitions |
| `data/scenarios.jsonl` | 30 scenarios — 20 clean, 10 adversarial |
| `slm/spec.py` | Behavior spec text — single source of truth for prompt and judge |
| `slm/scenarios.py` | Scenario model, loading, stratified sampling |
| `slm/providers.py` | Transport: Anthropic and OpenAI-compatible clients |
| `slm/prompting.py` | The three prompting strategies |
| `slm/checks.py` | Deterministic behavioral check |
| `slm/judge.py` | LLM-as-judge scoring |
| `slm/reporting.py` | Aggregation and results table |
| `ablation.py` | Models under test, the sweep, CLI |
