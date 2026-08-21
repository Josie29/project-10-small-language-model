# v1 vs v2 - the data change

v2 = v1 plus cross-paired rows that break the shape/concept confound. Training
hyperparameters and N are identical at every point, so the data is the only variable.
Regenerate with `python compare.py`.

## Probe: shape-swap generalisation

`cross-seen` are pairings v2 trained on. `cross-unseen` were withheld from training
on purpose - they are the test of whether the model learned to read the concept off
the code rather than memorising four more templates.

| N | control v1 | control v2 | cross-seen v1 | cross-seen v2 | cross-unseen v1 | cross-unseen v2 |
| ---: | --- | --- | --- | --- | --- | --- |
| 62 | 62% (5/8) | **50% (4/8)** -12% | 50% (2/4) | **75% (3/4)** +25% | 50% (2/4) | **50% (2/4)** 0 |
| 125 | 88% (7/8) | **62% (5/8)** -25% | 0% (0/4) | **75% (3/4)** +75% | 0% (0/4) | **50% (2/4)** +50% |
| 250 | 88% (7/8) | **88% (7/8)** 0 | 0% (0/4) | **75% (3/4)** +75% | 25% (1/4) | **75% (3/4)** +50% |
| 500 | 88% (7/8) | **100% (8/8)** +12% | 50% (2/4) | **75% (3/4)** +25% | 50% (2/4) | **75% (3/4)** +25% |

### Pooled over N=125/250/500

Individual cells hold 4 trials, so one judge flip moves a cell 25 points. These
pooled figures are the ones to read.

| Arm | v1 | v2 | delta |
| --- | --- | --- | ---: |
| control | 88% (21/24) | **83% (20/24)** | -4% |
| cross-seen | 17% (2/12) | **75% (9/12)** | +58% |
| cross-unseen | 25% (3/12) | **67% (8/12)** | +42% |

## 36-scenario eval set: no-regression check

| N | adherence v1 | adherence v2 | robustness v1 | robustness v2 |
| ---: | --- | --- | --- | --- |
| 62 | 38% (9/24) | **50% (12/24)** +12% | 67% (8/12) | **67% (8/12)** 0 |
| 125 | 79% (19/24) | **92% (22/24)** +12% | 67% (8/12) | **83% (10/12)** +17% |
| 250 | 100% (24/24) | **96% (23/24)** -4% | 100% (12/12) | **100% (12/12)** 0 |
| 500 | 100% (24/24) | **96% (23/24)** -4% | 100% (12/12) | **100% (12/12)** 0 |
