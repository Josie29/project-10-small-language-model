# Small Language Model

Project 10 of the Gauntlet AI program. Full requirements: [BRIEF.md](BRIEF.md).

**Behavior Spec:** given a short Python program with one mutable-state lifetime bug, the
model identifies the relevant declaration, assignment, or mutation and asks exactly one
non-compound question about creation, ownership, reset, or aliasing. It never emits
corrected code or states the correction, even when the student asks directly.

Full spec with edge-case rulings and metric definitions: [docs/behavior-spec.md](docs/behavior-spec.md).

## Status

Architecture Defense passed. Spec finalized, 36-scenario set written, prompt-ceiling
ablation harness built and **the full sweep has run** — 216 trials in
`results/state-lifetime-v1/`.

Prompting plateaus below the reliability bar. Best cell is 71% spec adherence / 67%
robustness (`claude-haiku-4.5`, zero-shot); no strategy on either model clears it. The
failure mode that survives every prompting attempt is **compound questions** — 77 of 94
violations are `multiple_questions`, where the model localizes correctly and then asks
two things at once. Per-concept, `ownership` is the weakest (0–56% across all six cells).

**The loop is closed, and then reopened by its own eval.** 500-example training pool, four
checkpoints on the data-efficiency curve, all public on the Hub, evaluated against the same
frozen judge and the same 36 held-out scenarios that produced the ablation numbers above.

Those checkpoints hit 100%/100% at N=250 — and a 16-scenario probe then showed that number
was measuring a shortcut, not the behavior. See [the confound](#the-confound-these-numbers-were-hiding)
and [the v2 data change](#v2-the-data-change) that recovers most of it.

Stack for the fine-tuning phase, including the QLoRA deviation:
[docs/tech-stack.md](docs/tech-stack.md).

## Base vs tuned

Same base model on both sides. The base is given the full behavior spec as a system
prompt — the strongest of the three strategies in the ablation. The tuned models are given
one line: *"You are a Python state-lifetime tutor."* The user turns are identical by
construction, so the system prompt is the only difference the delta can be attributed to.

| Model | N | Spec adherence | Robustness | Mechanical pass |
| --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-0.6B` (base, full spec prompt) | — | 0% | 0% | 0% |
| best prompted frontier model (`claude-haiku-4.5`) | — | 71% | 67% | 11% |
| tuned `…-n62` | 62 | 38% | 67% | 83% |
| tuned `…-n125` | 125 | 79% | 67% | 100% |
| tuned `…-n250` | 250 | **100%** | **100%** | 100% |
| tuned `…-n500` | 500 | **100%** | **100%** | 97% |

Raw per-example judge transcripts: `results/base-vs-tuned/trials.jsonl`. Pinned eval-code
commit, judge, and model revisions: `results/base-vs-tuned/run.json`.

The revisions in `run.json` are the **weights commits** — the exact ones these numbers were
measured against. `results/checkpoints.jsonl` records each repo's current HEAD, which is one
commit later because the model cards were written after the eval that fills them in. The
weights are byte-identical across that pair; `run.json` is the one to pin against.

## Data-efficiency curve

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/base-vs-tuned/curve-dark.png">
  <img alt="Spec adherence and robustness against training-set size, on a log x-axis. Both metrics rise from 38% adherence and 67% robustness at N=62 to 100% on both at N=250 and stay flat to N=500. Adherence crosses the 80% reliability bar between N=125, where it reaches 79%, and N=250." src="results/base-vs-tuned/curve-light.png">
</picture>

Regenerate with `python plot_curve.py`; it reads `results/base-vs-tuned/trials.jsonl`, so
the figure cannot drift from the numbers.

**Spacing.** Log-spaced by repeated halving from the full pool — 500 / 250 / 125 / 62.
Halving is the right step because the question is *order of magnitude*, not precision:
linear spacing would have spent all four runs in a region the behavior had already
saturated. Every point is a strict rank-ordered prefix of the one above it
(`slm/dataset.py` assigns ranks so each prefix stays balanced across concept, code shape,
and the clean:adversarial ratio), so the subsets are nested and **N is the only variable
between points** — not which examples happened to be drawn.

### Minimum viable N

**250**, with an honest asterisk on 125.

The bar is 80% clean adherence, set in [docs/tech-stack.md](docs/tech-stack.md) — chosen
because merely matching the 71% prompted ceiling would prove nothing. 250 clears it
outright at 100%. **125 lands at 79% — 19 of 24 — one scenario short.**

An earlier version of this section claimed 125, measured before the numbers were repinned
against exact Hub revisions. Rather than re-run until it reads 80%, the honest statement is
that 125 sits *on* the bar and is within judge noise of it: re-running the identical eval
against identical weights flips verdicts at about 3%, and one scenario here is 4 points. A
minimum-N claim that rests on a single scenario at a hand-picked threshold was never solid
enough to headline, in either direction.

500 buys nothing measurable over 250 on this eval set.

The honest reading of the low end: at N=62 the run is only 24 optimizer steps, because
epochs are held fixed across every point so that N stays the sole variable. The 38% at
that point is data *and* step starvation together, not sample efficiency alone.

### What the curve actually diagnosed

The failure mode that dies as N grows is **`wrong_lifetime_focus`** — 16 at n=62, 8 at
n=125, 0 from n=250 — not the format violations that capped the frontier models. Format
discipline is essentially free: even n=62 passes the mechanical check 83% of the time
against the best prompted frontier model's 11%.

**It is not localization that needs data.** An earlier version of this section said it was,
reading the judge's label at face value. The transcripts say otherwise: **23 of the 24
`wrong_lifetime_focus` failures quoted the correct bug region.** What fails is the
*question* — right line, wrong probe, asking when a set was *created* about a bug that is
about when it is *reset*. Chasing that down is what surfaced the shape/concept confound
below.

That also contradicts the escalation tripwire in `docs/tech-stack.md`, which reads dominant
`wrong_lifetime_focus` as a capacity problem calling for Qwen3-1.7B. It was a data problem.
The tripwire did not fire because it is gated on clean adherence under 80%, which never
happened past n=62.

## The confound these numbers were hiding

Every table above is measured on an eval set drawn from the same taxonomy as the training
data — and in that taxonomy **all 20 code shapes map to exactly one lifetime concept**
(`slm/generation.py` builds cells from `SHAPES_BY_CONCEPT[concept]`). A model can therefore
score 100% by learning `code shape → question template` without ever learning the concept.

`data/probe-shape-swap.jsonl` tests exactly that: 16 fresh scenarios, half pairing a
familiar shape with a concept v1 never gave it (**cross**), half keeping the home pairing on
equally unseen code (**control**). The control arm is what rules out plain novelty.

| checkpoint | control | cross |
| --- | ---: | ---: |
| n-125 | 88% (7/8) | 0% (0/8) |
| **n-250** | **88% (7/8)** | **12% (1/8)** |
| n-500 | 88% (7/8) | 50% (4/8) |

The checkpoint that scores 100%/100% above scores **12%** when the shape stops predicting
the concept, while the control arm holds at 88% on equally unseen code. Full transcripts in
`results/probe-v1/`, pinned to the exact Hub revisions and eval commit.

## v2: the data change

`data/train-v2/` is v1 plus 60 cross-paired rows, which cuts concept-pure shapes from
**20/20 to 8/20**. Every hyperparameter and every N is unchanged, so the data is the only
variable. Three pairings were **withheld from training on purpose** so the probe keeps an
unseen arm — the test of whether the fix generalises or just adds templates.

Pooled over N=125/250/500:

| Arm | v1 | v2 | delta |
| --- | --- | --- | ---: |
| control | 88% (21/24) | 83% (20/24) | −4 pts |
| cross-seen | 17% (2/12) | **75% (9/12)** | **+58 pts** |
| cross-unseen | 25% (3/12) | **67% (8/12)** | **+42 pts** |

The withheld pairings improved almost as much as the trained ones, which is the result that
matters: v2 taught the model to read the concept off the code, not four more templates.

**It is a trade, not a free win.** v2 costs about one scenario at the top of the
in-taxonomy curve — n-250 and n-500 adherence both go 100% → 96% — and the probe's control
arm is one trial down. In exchange the cross arm roughly triples. That is the right trade
for a spec that claims a *behavior*, but it should be read as a trade.

Full tables with raw counts: `results/delta-v1-v2.md` (`python compare.py`). Reasoning and
caveats: [docs/brainlift.md](docs/brainlift.md).

**Read the pooled rows, not the cells.** Each per-N cell holds 4 trials, so a single judge
flip moves a cell 25 points.

### Honest caveats

- **The eval set is 36 scenarios.** 100% on 24 clean examples is 100% of a small sample.
  Independently verified as uncontaminated (0 exact code collisions against the 500-example
  pool, highest shingle similarity 0.20, median 0.0), but the staff held-out set is the
  real test of whether the synthetic distribution covers the real one.
- **The tuned model learned a template.** 35 of 36 n-500 responses open with "Look at",
  mean length 75 characters. For this spec the template *is* the target behavior, but it is
  a reason to expect a drop on out-of-taxonomy input rather than to expect 100% to hold.
- **The base model scores 0%, which deserves scrutiny.** Its answers are not empty or
  malformed — they average 615 characters of fluent, confident, wrong explanation. A 0.6B
  model lectures instead of asking, and gets the mechanism wrong while doing it. That is
  the real starting point, not a rigged baseline.
- **This is LoRA on a bf16 base, not 4-bit QLoRA.** See the deviation note in
  [docs/tech-stack.md](docs/tech-stack.md).

## The training set

500 examples, balanced across a 40-cell grid of `lifetime_concept × code_shape ×
category`, with a `seed_domain` axis on top so the student learns the shape rather than
the variable names. 332 clean / 168 adversarial, matching the eval set's 2:1 ratio;
125 per concept; 500 distinct programs and 500 distinct student messages.

Every example carries its scenario *and* its on-spec response, authored together against
a known cell — the lab's "label by construction, not post-hoc judging" principle. That
removes the per-example judge call which dominated the cost of a naive design.

**Two authoring paths, one gate.** The v1 pool was authored in-session and ingested with
`--ingest`; `generate.py --target N` produces the same rows from a teacher model. Both go
through identical AST validation, eval-set contamination checking, mechanical screening,
dedupe, and rank assignment, and every row records which path produced it in
`provenance.author`. **The shipped rows were not produced by the script** — see the
caveats below.

| Check | Result |
|---|---|
| Eval-set contamination | 0 of 500 |
| Frozen-judge pass rate on a random 10% sample | 50 / 50 (100%) |
| Near-misses correctly rejected by the mechanical gate | 500 / 500 |

The audit uses the same frozen judge that scored the best prompted frontier model at 71%
adherence. It is reporting-only and never filters.

### The data-efficiency curve

Curve points are **id manifests**, not copies, and every prefix of the pool is balanced by
construction — so the subsets are automatically nested and each point differs only in N:

| Point | Clean share | Concepts | Code shapes | Domains |
|---|---:|---:|---:|---:|
| `n-62` | 67.7% | 4/4 | 20/20 | 12/12 |
| `n-125` | 68.0% | 4/4 | 20/20 | 12/12 |
| `n-250` | 66.0% | 4/4 | 20/20 | 12/12 |
| `n-500` | 66.4% | 4/4 | 20/20 | 12/12 |

Ranks are never renumbered, so growing the pool later leaves these four points untouched.

### Honest caveats

- **The v1 rows were authored in-session, not generated by `generate.py`.** The script
  implements the same taxonomy and the same gate and is smoke-tested against a live
  teacher (`data/train-smoke/`), but it did not produce the shipped rows. Every row says
  so in `provenance.author`.
- **Single-context authoring is a real diversity risk.** The lab's diversity mechanism is
  independent high-temperature samples; one context has neither. Mitigations: cell-by-cell
  authoring, an explicit domain per example, and normalized-code dedupe. It is measurable
  — the eval set is independent, so thin coverage shows up as an early-flattening curve.
- **The training filter is deliberately stricter than the eval judge.** The mechanical
  check flags any `and`/`or` in a question clause. That over-rejects some legitimate
  phrasing, which is the point: `multiple_questions` was 77 of 94 frontier violations.
- **Label-by-construction means the gate rejects little.** The filtering claim rests on
  the AST, contamination, and mechanical checks plus the audit number — not on a high
  reject count.

## Running data generation

```bash
python generate.py --dry-run                    # no API calls, proves the pipeline
python generate.py --ingest <batch.jsonl>       # in-session authored rows through the gate
python generate.py --target 500                 # teacher path, ~$2 with Sonnet 5
python generate.py --audit --audit-frac 0.10    # judge a sample, reporting only
```

Writes `data/train/pool-v1.jsonl` (canonical pool), `curve/n-*.txt` (manifests),
`sft-v1.jsonl` (TRL-ready chat format), `raw/` (every candidate plus rejections, which
double as DPO preference pairs), and `audit-v1.jsonl` (judge transcripts).

## Running training, eval, and publishing

```bash
uv sync --extra train                 # torch, transformers, trl, peft, at locked versions
cp .env.example .env                  # OPENROUTER_API_KEY for the judge, HF_USER for the Hub

python train.py --dry-run             # slices the pool, renders configs, loads no weights
python train.py --sizes 62,125,250,500

python eval.py --dry-run              # no network: prompts, checks, both table renderers
python eval.py --model <hf-repo-id>   # base vs one checkpoint on the 36-scenario eval set

python publish.py --dataset --models  # Hub push, cards, pinned shas
```

`train.py` runs on Apple Silicon by default (`--backend local`, MPS) and takes about 25
minutes for all four curve points. `--backend modal` runs the identical recipe on an A10G
with a 4-bit base; see the QLoRA deviation in [docs/tech-stack.md](docs/tech-stack.md).

`eval.py` loads each checkpoint in-process, so reproducing the table needs no GPU and no
serving account — about 20 minutes on a laptop for the base plus four checkpoints. It
writes `results/base-vs-tuned/`: `trials.jsonl` (per-example judge score and reasoning),
`table.md`, `curve.md`, and `run.json` pinning the eval-code commit, the judge, and every
model revision.

## Running the prompt-ceiling ablation

```bash
uv sync                       # creates .venv and installs from uv.lock
cp .env.example .env          # then fill in OPENROUTER_API_KEY; .env is gitignored

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
| `docs/tech-stack.md` | Stack decisions for the fine-tuning phase, with the base-model tripwire |
| `results/state-lifetime-v1/` | Prompt-ceiling ablation output — raw judge transcripts and results table |
| `data/scenarios.jsonl` | 36 state/lifetime **eval** scenarios — 24 clean, 12 adversarial. Never trained on. |
| `data/probe-shape-swap.jsonl` | 16 shape-swap probe scenarios — 8 cross-paired, 8 home-paired controls. Never trained on. |
| `data/heldout-union.jsonl` | Both eval sets concatenated; what `generate.py --eval-set` checks contamination against |
| `data/train/` | The 500-example v1 pool, curve manifests, SFT export, raw batches, audit |
| `data/train-v2/` | The 560-example v2 pool. `pool-v1.jsonl` inside it is the *v2* pool — the filename is a code constant, not a version. Its `curve/` manifests are halvings of 560 (70/140/280/560) and were **not** the points trained: v2 was trained at 62/125/250/500 to match v1 exactly, selected by rank prefix via `curve_subset`, which never reads the manifests. |
| `data/train-smoke/` | Live smoke-test output proving the teacher path works end to end |
| `slm/spec.py` | Behavior spec text — single source of truth for prompt and judge |
| `slm/scenarios.py` | Scenario model, loading, stratified sampling |
| `slm/generation.py` | Taxonomy (code shapes, pressures, domains), the 40-cell grid, teacher prompt |
| `slm/dataset.py` | Training row model, AST validation, contamination check, gate, ranking, export |
| `generate.py` | Generation and ingest CLI |
| `slm/providers.py` | Transport: Anthropic and OpenAI-compatible clients |
| `slm/prompting.py` | The three prompting strategies |
| `slm/checks.py` | Deterministic behavioral check |
| `slm/judge.py` | LLM-as-judge scoring |
| `slm/reporting.py` | Aggregation, results table, data-efficiency curve |
| `slm/sft.py` | The trainer — LoRA/QLoRA, prompt masking, merge |
| `slm/local.py` | In-process `transformers` provider for the student checkpoints |
| `slm/training.py` | Run config, curve subsetting, SFT export |
| `slm/checkpoints.py` | Checkpoint manifest with pinned Hub revisions |
| `slm/publishing.py` | Hub pushes, model and dataset cards |
| `ablation.py` | Models under test, the sweep, CLI |
| `train.py` | Training sweep CLI — local MPS or Modal |
| `eval.py` | Base-vs-tuned eval CLI |
| `publish.py` | Dataset, model card, and demo publishing CLI |
| `plot_curve.py` | Renders the data-efficiency curve from the trial records |
| `compare.py` | Renders the v1-vs-v2 delta tables, splitting the probe's seen and withheld arms |
| `docs/brainlift.md` | Behavior thesis, the confound, and whether data → behavior held |
| `modal_app.py` | Modal A10G image and entrypoint for a 4-bit rerun |
| `space/` | Gradio demo, base vs tuned, containerised for CPU hosting |
| `results/base-vs-tuned/` | Base-vs-tuned trials, table, curve, and pinned run manifest |
| `results/probe-v1/`, `results/probe-v2/` | Shape-swap probe trials for the v1 and v2 checkpoints |
| `results/delta-v1-v2.md` | The v1-vs-v2 comparison, rendered by `compare.py` |
| `results/train/`, `results/train-v2/` | Per-checkpoint training logs — `trainer_state.json` and loss CSV |
| `results/checkpoints.jsonl` | Every published checkpoint with its Hub commit sha |
| `tests/` | Regression tests for the state/lifetime checks and scenario balance |
