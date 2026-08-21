# Brainlift

Every number here comes from one pinned sweep: eval code at commit `3b637d1` (base-vs-tuned
at `d4418b6`), the frozen judge, and the exact Hub revisions recorded in each
`results/*/run.json`. Regenerate the comparison tables with `python compare.py`.

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
| 62 | 38% | 67% |
| 125 | 79% | 67% |
| 250 | **100%** | **100%** |
| 500 | **100%** | **100%** |

Read on its own this says the thesis held completely: 250 examples turn a 0.6B model that
scores 0% into one that never misses, beating the 71% frontier prompting ceiling outright.

That reading is wrong.

### The failure mode I diagnosed from the MVP eval

The judge labelled 24 of the 28 v1 failures `wrong_lifetime_focus`, and my first pass at
the README concluded from that label that *localization* was the thing data buys. It is
not. **23 of those 24 responses quoted the correct bug region.** Localization was already
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

My MVP answer was **125**, the smallest point clearing an 80% clean-adherence bar. Two
separate things undermined that number, in opposite directions — and then the v2 data
change landed on 125 again for a completely different reason. The route matters more than
the destination here, so it is worth walking.

**It was never robust to a rerun.** Repinned against exact Hub revisions, v1's n-125 scores
**79% — 19 of 24, one scenario short of the bar it was chosen for.** Nothing about the model
changed; the same weights were re-judged. Judge verdicts flip at about 3% run to run and one
scenario here is 4 points, so 125 was always sitting *on* the threshold rather than above it.
A minimum-N claim resting on one scenario at a hand-picked bar is not a finding, it is a
coin landing on its edge.

**And the bar was measuring the wrong distribution anyway.** Whatever 125 scores, it scores
it *on an eval set that shares the training taxonomy*. On the cross arm the same checkpoint
scores 0% (0/8). A minimum viable N is only meaningful relative to the distribution you
intend to hold the behavior over, and my MVP answer silently assumed that distribution was
the one I had generated.

Against the probe, no checkpoint at any N clears 80% on the cross arm — v2's best is 75% (6/8)
at n=250 and n=500. So the honest restatement is:

- **125 is the minimum viable N, on v2, over the training taxonomy.** v1 needs 250 to clear
  the same bar. **The data change halved the minimum viable dataset size.**
- **There is no N in this sweep at which the behavior holds reliably off-taxonomy.** v2
  more than triples cross-arm performance (21% → 71% pooled) but does not reach the bar.
  Claiming a minimum viable N for the general behavior would repeat exactly the mistake v1
  made.

| N | v1 adherence | v2 adherence |
| ---: | --- | --- |
| 62 | 38% (9/24) | 50% (12/24) |
| 125 | **79% (19/24)** | **92% (22/24)** |
| 250 | 100% (24/24) | 96% (23/24) |
| 500 | 100% (24/24) | 96% (23/24) |

The reason to state it that way is that **minimum viable N is a property of the dataset,
not the model.** Nothing differs between those two columns except which rows are in the
pool — same base, same rank, alpha, LR, schedule, epochs and masking, same 36 scenarios,
same frozen judge. So the halving measures the data change directly, and it measures the
same thing the cross-arm gains measure, from the other end. The mechanism is the one the
probe exposed: v1 at 125 is still fitting `shape → question template`, and 20 shapes do not
fit into 125 examples. v2 cannot use that route for 12 of the 20 shapes, so it has to read
the concept off the code — and a concept transfers across shapes in a way a template does
not. Breaking the shortcut did not just improve generalisation, it improved sample
efficiency, and I did not predict that.

It is worth being explicit that **the lower number is the better-evidenced one**, which is
backwards from the usual direction of this claim. My 250 answer rested on 125 missing by one
scenario at a hand-picked bar — a coin landing on its edge, as above. v2's 125 is three
scenarios clear of it, and its robustness at that point is 83% against v1's 67%.

**And then I applied lesson 5 to my own headline.** Adherence is 24 clean scenarios, so one
scenario is 4.2 points and the 80% bar cannot actually be scored — it falls between 19/24
(79.2%) and 20/24 (83.3%). Worse, at n=24 the 95% Wilson interval on v2's 91.7% is
**[74%, 98%]**: the point estimate clears the bar and *the interval does not*. It would take
about 48 clean scenarios for a 92% observation to put its lower bound above 80%.

I am not going to grow the eval set to fix that. It is contamination-checked and every
committed number is pinned against it, so expanding it on the last day would invalidate the
comparison it exists to support, and I would be choosing a bigger n over a stable one at
exactly the moment I have the least time to verify it. What I will do is stop quoting the
absolute number as if it were sharp. **The halving is the robust claim, not the 125.** It is
a relative comparison — same 24 scenarios, same judge, same weights recipe, only the pool
differs — so the sampling error that makes the absolute number soft largely cancels. "v2
reaches the bar at half the data v1 needs" survives a sample size that "v2 achieves >80% at
N=125" does not.

Minimum viable N is not the same question as which checkpoint to submit. That is v2-n500:
the demo takes arbitrary input, and n-500 has the best cross-arm rate.

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
   was fine in 23 of 24 cases. The label pointed at the wrong half of the response.
5. **A threshold result is not a result.** My minimum-N answer was 125 because 125 scored
   88% against an 80% bar. Repinning the same weights moved it to 79%. Nothing changed but
   the judge's coin flips. If a headline number would flip on one example, report the
   interval, not the number.
6. **Pin the demo's dependencies before you need the demo.** `space/requirements.txt` had
   `gradio>=5.0` with no upper bound. Redeploying to swap one model id pulled Gradio 6,
   which had removed a kwarg the app passed, and the live demo went to 502 — a build that
   had worked unchanged for days. An unpinned range means the demo you rehearse is not the
   demo the grader loads.

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
