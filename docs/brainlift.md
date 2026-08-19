# Brainlift — in progress

Status: draft for Early Submission. The v1 and probe sections are final; the v2 section
is filled in as the retrained checkpoints land.

## The behavior thesis

Given a short Python program with one mutable-state lifetime bug, the model identifies
the relevant declaration, assignment, or mutation and asks exactly one non-compound
question about creation, ownership, reset, or aliasing. It never emits corrected code or
states the correction, even when the student asks directly.

The bet: this is a *reliability* behavior, not a capability one. A frontier model knows
perfectly well what a mutable-default bug is. What it cannot do reliably is hold the
constraint — one question, no answer — turn after turn under pressure. The prompt-ceiling
ablation confirmed the bet before any training code was written: 216 trials, two model
families, three strategies, best cell 71% adherence / 67% robustness, and 77 of 94
violations were the same failure — compound questions.

## Did data → behavior hold?

**Yes on the axis I varied, and no on the axis I held fixed.** That distinction is the
whole finding, and I did not see it until I built an eval that could.

### What v1 showed

Four checkpoints on a log-spaced curve, evaluated on 36 held-out scenarios:

| N | Spec adherence | Robustness |
| ---: | ---: | ---: |
| 62 | 46% | 67% |
| 125 | 88% | 75% |
| 250 | **100%** | **100%** |
| 500 | **100%** | **100%** |

Read on its own this says the thesis held completely: 250 examples turn a 0.6B model that
scores 0% into one that never misses, beating the 71% frontier prompting ceiling outright.

That reading is wrong.

### The failure mode I diagnosed from the MVP eval

The judge labelled 21 of the 23 v1 failures `wrong_lifetime_focus`, and my first pass at
the README concluded from that label that *localization* was the thing data buys. It is
not. **20 of those 21 responses quoted the correct bug region.** Localization was already
free. What failed was the question:

| concept | what the model asked |
| --- | --- |
| reset | "When does that set get **created**?" |
| aliasing | "Which list does that list **create**?" |
| ownership | "When does that list get **emptied**?" |

Right line, wrong probe. Chasing that down produced the actual defect — and it was in my
data, not my model.

### The defect: shape and concept were perfectly collinear

`slm/generation.py` builds its 40-cell grid from `SHAPES_BY_CONCEPT[concept]`, which means
each of the 20 code shapes belongs to exactly one lifetime concept. Measured on the
shipped pool:

```
20/20 code shapes map to exactly ONE concept
list_multiplication → aliasing 100%    accumulator_in_loop → reset 100%
plain_assignment    → aliasing 100%    class_attribute     → ownership 100%
```

I had checked balance on every axis I could think of — concept, code shape, seed domain,
clean:adversarial ratio, and every prefix of the curve. I never checked the *interaction*.
Because shape determined concept with 100% accuracy, the model never had to learn the
concept at all. It could learn `syntax → question template` and score 100%.

And my eval set could not detect that, because the eval set is drawn from the same
taxonomy. Train and eval shared the confound, so the eval certified the shortcut.

### The probe that caught it

`data/probe-shape-swap.jsonl` — 16 fresh scenarios in two arms:

- **cross** (8): a familiar code shape carrying a concept v1 never paired it with.
- **control** (8): a familiar code shape carrying its home concept, on code equally unseen.

The control arm is what makes it evidence rather than a vibe. A drop on cross alone could
be explained by novelty; a drop on cross while control holds isolates the confound.

| checkpoint | control | cross | gap |
| --- | ---: | ---: | ---: |
| n-62 | 62% (5/8) | 50% (4/8) | 12 pts |
| n-125 | 88% (7/8) | 0% (0/8) | 88 pts |
| **n-250** | **88% (7/8)** | **12% (1/8)** | **75 pts** |
| n-500 | 88% (7/8) | 50% (4/8) | 38 pts |

The checkpoint that scores 100%/100% on my own eval set scores **12%** when the shape stops
predicting the concept. Every cross-arm failure asked the question belonging to the
shape's v1 home concept — six of eight are a clean substitution:

| scenario | shape (home → actual) | model asked |
| --- | --- | --- |
| cross-01 | list_multiplication (aliasing → **reset**) | "Which list does that expression build?" |
| cross-02 | accumulator_in_loop (reset → **aliasing**) | "When does that dictionary start over?" |
| cross-03 | class_attribute (ownership → **reset**) | "Which Report objects hold that list?" |

Note the gap *grows* with N through 250. More in-taxonomy data makes the shortcut
stronger. That is the signature of shortcut learning, not of undertraining, and it is the
single most useful number in this project.

## The v2 data change

v1 + 60 cross-paired rows spanning 12 shapes, so that shape stops determining concept:
**20/20 concept-pure shapes → 8/20**. No hyperparameter moved; `--sizes` are matched to
v1 exactly (62 / 125 / 250 / 500) so N is held fixed and the data is the only variable.

Three pairings are **deliberately withheld** from v2 training so the probe keeps an unseen
arm:

```
accumulator_in_loop → aliasing     (probe cross-02, cross-08)
shallow_copy        → reset        (probe cross-04)
counter_reinit      → ownership    (probe cross-05)
```

This is the part that decides whether the data change taught a behavior or four more
templates. If only the seen arm recovers, v2 memorised the pairings I showed it. If the
withheld arm recovers too, the model is reading the concept off the code — which is the
claim the spec actually makes.

### What v2 did

Pooled over N=125/250/500 (`results/delta-v1-v2.md`, regenerate with `python compare.py`):

| Arm | v1 | v2 | delta |
| --- | --- | --- | ---: |
| control | 88% (21/24) | 83% (20/24) | −4 pts |
| cross-**seen** | 17% (2/12) | **75% (9/12)** | **+58 pts** |
| cross-**unseen** | 25% (3/12) | **67% (8/12)** | **+42 pts** |

**The withheld pairings improved too.** That is the result the hold-out was built to test.
If v2 had merely memorised the three pairings I showed it, the unseen arm would have sat
still; instead it gained 42 points, and it gained at every one of N=125, 250 and 500
individually (+50, +50, +25). The model is reading the concept off the code rather than off
the syntax — partially, not perfectly.

The seen arm gains more than the unseen arm (+58 vs +42), which is the honest shape of the
result: direct supervision on a pairing helps more than transfer to a withheld one. Both
matter, and reporting only the seen number would have been the "teaching to the test"
failure this design was built to avoid.

**What it cost.** This is a trade, not a free win. On the 36-scenario in-taxonomy eval,
n-250 and n-500 adherence each slip one scenario (100% → 96%), and the probe's control arm
is one trial down (88% → 83%). Each is a single-trial movement at the edge of what this
sample can resolve, but they point the same way, and the honest summary is that v2 buys a
large cross-taxonomy gain for a small in-taxonomy cost. n-125 moved the other way
(adherence 88% → 92%, robustness 75% → 83%).

For a spec that claims a *behavior* rather than a benchmark score, that trade is worth
taking — but it is a trade. An earlier draft of this section said "nothing meaningful
regressed", which was true of the run it was written against and is not true of the pinned
one reported here.

At n=62 both arms got worse. That is expected and was predicted before the run: only 2 of
62 rows are cross-paired at that prefix, so v2 ≈ v1 there by construction, and the
movement is 1–2 scenarios of noise on a checkpoint that trains for 24 optimizer steps.

## Minimum viable N

v1's stated answer was **125**, the smallest point clearing an 80% clean-adherence bar.

That number now needs an asterisk. 125 clears the bar *on an eval set that shares the
training taxonomy*. On the cross arm the same checkpoint scores 0% (0/8). A minimum viable N is
only meaningful relative to the distribution you intend to hold the behavior over, and my
v1 answer silently assumed that distribution was the one I had generated.

Against the probe, no checkpoint at any N clears 80% on the cross arm — v2's best is 75% (6/8)
at n=250 and n=500. So the honest restatement is:

- **125 remains the minimum viable N for the behavior as specified over the training
  taxonomy.** v2 does not change that; it holds there at 92% adherence, up from v1's 88%.
- **There is no N in this sweep at which the behavior holds reliably off-taxonomy.** v2
  more than triples cross-arm performance (21% → 71% pooled) but does not reach the bar.
  Claiming a minimum viable N for the general behavior would repeat exactly the mistake v1
  made.

That is a less satisfying headline than "100% at 250" and it is the one the evidence
supports.

## What I would tell someone starting this project

1. **Balance is not the same as independence.** I checked every marginal distribution and
   every one of them was clean. The defect was in a two-way interaction that no marginal
   could reveal. If your generator enumerates cells as `for concept: for shape in
   SHAPES[concept]`, your axes are nested, not crossed.
2. **An eval drawn from the training taxonomy cannot detect a taxonomy-shaped bug.** Mine
   certified a shortcut at 100%. The cheapest insurance is a small probe that deliberately
   violates one structural assumption of the generator — 16 scenarios and one afternoon
   found what 36 scenarios and four checkpoints could not.
3. **Always build the control arm.** Without the 8 home-paired controls the cross-arm
   collapse would have been arguable as "it's just unfamiliar code." With them it is not
   arguable.
4. **Read the judge's label, then check it.** `wrong_lifetime_focus` sounds like a
   localization failure and I wrote that into my README. The transcripts said localization
   was fine in 20 of 21 cases. The label pointed at the wrong half of the response.

## Honest caveats

- **The probe is 16 scenarios**, and each per-N cell holds 4. It is a smoke detector, not a
  benchmark — powerful because the arms are matched, not because the sample is large. Quote
  the pooled rows, not the cells.
- **The judge is not deterministic, and that is the largest source of noise here.**
  Re-running the identical probe against identical weights flipped 2 of 63 verdicts (3%)
  with byte-identical model responses on all 63 — the variance is the judge, not sampling.
  On a 4-trial cell one flip is 25 points, which is why only the pooled rows are quoted.
  Every number in this document comes from one pinned sweep at eval commit `3b637d1`
  against the exact Hub revisions in each `run.json`; `results/probe-v2-run1/` keeps a
  superseded earlier run for comparison. The pinned numbers are *not* uniformly the
  flattering ones — the control arm reads −4 points here where an earlier run read +4.
- **A judge failure was silently costing trials.** The judge intermittently degenerates
  into blank lines, hits the token cap, and returns JSON truncated mid-object; `eval.py`
  logged a warning and dropped the trial, so a graded artifact quietly lost evidence. It
  reproduced at roughly one call in three on one probe scenario. `judge_response` now
  retries before failing loudly. Every run reported here is complete — 64, 64 and 144
  trials with zero drops.
- **v2's cross-paired rows are ~11% of every curve prefix**, and only 2 rows at n=62. The
  data change is expected to bite at 125 and above; the low end is essentially unchanged
  by construction, and any n=62 delta should be read as noise.
- **12 of 20 shapes are now cross-paired; 8 remain concept-pure.** The confound is broken,
  not eliminated. A grader could build the same probe against the remaining 8 shapes.
- **These rows were authored in-session, like v1's**, through the identical gate. The
  single-context diversity risk carries over. `generate.py --target` implements the same
  taxonomy and gate but did not produce the shipped rows.
- **This is LoRA on a bf16 base, not 4-bit QLoRA** — bitsandbytes has no Metal backend.
  Same rank, alpha, targets, LR, schedule, and masking; `--backend modal` reruns it 4-bit.
