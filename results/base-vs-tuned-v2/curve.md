# Data-Efficiency Curve

Spec adherence = judge pass rate on clean scenarios.
Robustness = judge pass rate on adversarial scenarios.

Epochs are held fixed across every point, so the smallest N also trains for the
fewest optimizer steps. Read the low end as data *and* step starvation together.

| N | Checkpoint | Spec adherence | Robustness | Mechanical pass |
| ---: | --- | ---: | ---: | ---: |
| 62 | `checkpoints/local-eval/tutor-n62-v2` | 50% | 67% | 86% |
| 125 | `checkpoints/local-eval/tutor-n125-v2` | 88% | 92% | 97% |
| 250 | `checkpoints/local-eval/tutor-n250-v2` | 100% | 100% | 100% |
| 500 | `checkpoints/local-eval/tutor-n500-v2` | 96% | 100% | 100% |
