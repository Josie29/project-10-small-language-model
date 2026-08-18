# Small Language Model

Project 10 of the Gauntlet AI program. Full requirements: [BRIEF.md](BRIEF.md).

**Behavior Spec:** given a short Python program with one mutable-state lifetime bug, the
model identifies the relevant declaration, assignment, or mutation and asks exactly one
non-compound question about creation, ownership, reset, or aliasing. It never emits
corrected code or states the correction, even when the student asks directly.

Full spec with edge-case rulings and metric definitions: [docs/behavior-spec.md](docs/behavior-spec.md).

## Status

Architecture Defense passed. Spec finalized, 30-scenario set written, prompt-ceiling
ablation harness built and **the full sweep has run** — results in `results/`.

The previous broad-code ablation is retained in `results/`. Run the focused state/lifetime
ablation before drawing conclusions from this new behavior.

## Running the prompt-ceiling ablation

```bash
uv venv && uv pip install anthropic openai pydantic
cp .env.example .env          # then fill in the three keys; .env is gitignored

python ablation.py --dry-run   # no API calls, proves the pipeline wiring
python ablation.py --limit 2   # smoke test: 24 model-and-judge calls
python ablation.py             # full sweep: 432 model-and-judge calls
```

Writes `results/state-lifetime-v1/trials.jsonl` (per-example judge score and reasoning
— the raw transcripts the brief requires) and `results/state-lifetime-v1/table.md`
(the results table). Pass `--out` to run a separately named experiment.

Models are configured in `MODELS_UNDER_TEST` in `ablation.py` and `JUDGE` in
`slm/config.py`. The judge is deliberately a third family so it never grades its own
output, and it is **frozen** — changing it invalidates comparison across runs.

## Layout

| Path | What |
|---|---|
| `docs/behavior-spec.md` | The spec, edge-case rulings, metric definitions |
| `data/scenarios.jsonl` | 36 state/lifetime scenarios — 24 clean, 12 adversarial |
| `slm/spec.py` | Behavior spec text — single source of truth for prompt and judge |
| `slm/scenarios.py` | Scenario model, loading, stratified sampling |
| `slm/providers.py` | Transport: Anthropic and OpenAI-compatible clients |
| `slm/prompting.py` | The three prompting strategies |
| `slm/checks.py` | Deterministic behavioral check |
| `slm/judge.py` | LLM-as-judge scoring |
| `slm/reporting.py` | Aggregation and results table |
| `ablation.py` | Models under test, the sweep, CLI |
| `tests/` | Regression tests for the state/lifetime checks and scenario balance |
