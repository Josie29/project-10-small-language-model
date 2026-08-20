# Demo script — Early Submission (Thursday), 4:00

The MVP demo pitched a number: 100% adherence at N=250. This one is about why that number
was wrong, how the eval that caught it was built, and what the data change bought. Early
Submission is a check-in, so the through-line is *iterating on data*, not a victory lap.

Every figure below is from the pinned sweep — eval commit `3b637d1`, base-vs-tuned at
`d4418b6`, exact Hub revisions in each `results/*/run.json`.

## Before you hit record

- [ ] Open <https://slm-state-lifetime-demo-production.up.railway.app> and click **Compare**
      once. Models load into RAM on first request (~15s); a cold start on camera is the one
      avoidable disaster.
- [ ] Tabs, in order: `results/delta-v1-v2.md` · `data/probe-shape-swap.jsonl` · the demo ·
      the n500-v2 model page on Hugging Face.
- [ ] The demo serves **base vs v2-n500**. It cannot show v1 — that contrast comes from the
      committed transcripts in step 3, not the live tool.
- [ ] Grader's prompt pasted somewhere unread until step 6.

## Flow

| # | Time | On screen | Beat |
|---|---|---|---|
| 1 | 0:00–0:25 | `results/base-vs-tuned/curve.md` | The number I stopped trusting |
| 2 | 0:25–1:05 | trials, then `slm/generation.py` | The diagnosis, and the defect under it |
| 3 | 1:05–1:50 | `data/probe-shape-swap.jsonl` | The probe, and why it has a control arm |
| 4 | 1:50–2:25 | `data/train-v2/` | The data change, and the deliberate hold-out |
| 5 | 2:25–3:10 | `results/delta-v1-v2.md` | What it bought, and what it cost |
| 6 | 3:10–3:50 | **Demo, side by side** | Live: cross-paired bug, then grader's prompt |
| 7 | 3:50–4:00 | — | Where it actually stands |

---

## 1 — The number I stopped trusting (0:25)

> At MVP I had four checkpoints and this table. N=250 scores **100% spec adherence, 100%
> robustness** on 36 held-out scenarios, against a prompted frontier ceiling of 71%.
>
> That is the number I spent this round proving wrong. Not the arithmetic — the claim.

## 2 — The diagnosis (0:40)

Show the failing trials, then `slm/generation.py:150`.

> The judge labels most failures `wrong_lifetime_focus`. I originally read that as *the
> model can't find the buggy line* and wrote that into my README.
>
> The transcripts say otherwise. **23 of the 24 mislocalisation failures quoted the correct
> bug region.** Localisation was already free. What fails is the *question* — right line,
> wrong probe. It asks when a set was **created** about a bug that is about when it is
> **reset**.

Now the generator.

> Chasing that down found the defect, and it is in my data. My grid is built as
> `for concept: for shape in SHAPES_BY_CONCEPT[concept]`. Those axes are nested, not
> crossed. Measured on the shipped pool: **all 20 code shapes map to exactly one lifetime
> concept.**
>
> So the model never has to learn the concept. It can learn `syntax → question template`
> and score 100%. And my eval set cannot see that, because it is drawn from the same
> taxonomy. Train and eval shared the confound, so the eval certified the shortcut.

## 3 — The probe (0:45)

> 16 fresh scenarios, two arms. **Cross**: a familiar code shape carrying a concept the
> training data never paired it with. **Control**: a familiar shape keeping its home
> concept, on code equally unseen.
>
> The control arm is the whole design. Without it, a drop is arguable as "that's just
> unfamiliar code." With it, the comparison is clean.

| checkpoint | control | cross |
|---|---:|---:|
| n-125 | 88% (7/8) | 0% (0/8) |
| **n-250** | **88% (7/8)** | **12% (1/8)** |
| n-500 | 88% (7/8) | 50% (4/8) |

> The checkpoint that scores 100% on my own eval set scores **12%** when the shape stops
> predicting the concept — while the control arm holds at 88% on equally new code.
>
> And the gap *widens* with N through 250. More in-taxonomy data makes the shortcut
> stronger. That is shortcut learning, not undertraining.

## 4 — The data change (0:35)

> v2 is v1 plus 60 cross-paired rows across 12 shapes. Concept-pure shapes drop from
> **20 of 20 to 8 of 20**. No hyperparameter moved and N is matched exactly at every point,
> so the data is the only variable.
>
> The part I want credit for: **three pairings are withheld from training on purpose** —
> `accumulator_in_loop→aliasing`, `shallow_copy→reset`, `counter_reinit→ownership`. They
> stay in the probe. If only the pairings I taught improve, I taught four more templates.
> If the withheld ones improve too, the model is reading the concept off the code.

## 5 — What it bought, and what it cost (0:45)

| Arm | v1 | v2 | delta |
|---|---|---|---:|
| control | 88% (21/24) | 83% (20/24) | −4 pts |
| cross-**seen** | 17% (2/12) | **75% (9/12)** | **+58** |
| cross-**withheld** | 25% (3/12) | **67% (8/12)** | **+42** |

> The withheld pairings gained 42 points. That is the result the hold-out was built to test.
>
> Say the cost out loud: this is a **trade**, not a free win. n-250 and n-500 each lose one
> scenario of in-taxonomy adherence, 100% to 96%, and the control arm is one trial down. I
> think that trade is right for a spec that claims a behavior rather than a benchmark score
> — but it is a trade.
>
> Read the pooled rows, not the cells. Each per-N cell is 4 trials, and re-running the
> identical eval flips verdicts at about 3%.

## 6 — Live (0:40) — the load-bearing beat

Paste this into the demo. It is a `list_multiplication` shape carrying a **reset** bug —
the exact pairing v1 gets wrong.

```python
def tally_rows(rows):
    counts = [0] * 3
    for row in rows:
        counts = [0] * 3
        counts[row] += 1
    return counts
```
> "Only the last row ends up counted."

Click **Compare**. Base on the left with the full spec prompt, v2 on the right with one line.

> The base lectures — hundreds of characters, confident, and wrong about the mechanism.
>
> v2 asks: *"What happens to the previous counts each time that line runs?"* That is a reset
> question about a reset bug.
>
> For contrast, here is what **v1** said on this same program, from the committed
> transcript: *"Which index does that list reach?"* It sees `[0] * 3`, which in its training
> data was always aliasing, and asks the aliasing question. Same line, wrong concept.

**Now the grader's prompt.** Paste unedited, click Compare, read the tuned output aloud.

*If the grader has nothing prepared, offer this — a `counter_reinit → ownership` bug, which
is one of the **withheld** pairings, so v2 has never trained on this combination:*

```python
class Counter:
    tally = {}

    def bump(self, key):
        self.tally[key] = self.tally.get(key, 0) + 1
        return self.tally
```
> "A second counter starts with the first one's numbers already in it."

## 7 — Where it stands (0:10)

> Minimum viable N is **250**, not the 125 I claimed at MVP — repinned, 125 scores 79%
> against an 80% bar. One scenario. I could have re-run until it read 80%; the honest
> version is that 125 sits *on* the bar.
>
> And no checkpoint at any N clears that bar off-taxonomy. v2 more than triples cross-arm
> performance, 21% to 71%, and still does not get there. That is the Sunday problem.

---

## Do not say

- **"It works."** Every claim here has a number and a transcript behind it; use them.
- **"100% adherence"** without immediately saying which distribution. That framing is the
  exact mistake this round exists to correct.
- **"Fixed."** v2 *reduces* the shortcut. 8 of 20 shapes are still concept-pure, and a
  grader could rebuild my own probe against those and get the same collapse.
- Do not call it QLoRA. It is LoRA on a bf16 base — bitsandbytes has no Metal backend. If
  asked: same rank, alpha, targets, LR, schedule and masking; only the frozen base's
  precision differs, and `--backend modal` reruns it 4-bit unchanged.
- Do not imply the probe or the eval set were trained on. Both are contamination-checked:
  0 exact collisions, max shingle overlap 0.13.

## If asked

- **"Why is the control arm down 4 points?"** One trial, 21/24 to 20/24, inside judge noise.
  I am not claiming v2 improved control — only that it did not break it.
- **"Isn't the probe too small?"** Yes, 16 scenarios, 4 per cell. It is a smoke detector,
  not a benchmark. It is convincing because the arms are matched, not because it is large —
  which is why I quote pooled rows.
- **"Did you tune hyperparameters?"** No. Same recipe, same epochs, same N at every point.
  The only change is which rows are in the pool.
- **"Which model would you submit?"** v2-n500. It gives up ~4 points in-taxonomy for roughly
  triple the cross-taxonomy pass rate, and the demo takes arbitrary input.
