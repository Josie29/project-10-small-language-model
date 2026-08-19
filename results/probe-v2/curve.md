# Data-Efficiency Curve

Spec adherence = judge pass rate on clean scenarios.
Robustness = judge pass rate on adversarial scenarios.

Epochs are held fixed across every point, so the smallest N also trains for the
fewest optimizer steps. Read the low end as data *and* step starvation together.

| N | Checkpoint | Spec adherence | Robustness | Mechanical pass |
| ---: | --- | ---: | ---: | ---: |
| 62 | `machalek29/qwen3-0.6b-state-lifetime-tutor-n62-v2` | 58% | 50% | 69% |
| 125 | `machalek29/qwen3-0.6b-state-lifetime-tutor-n125-v2` | 58% | 75% | 88% |
| 250 | `machalek29/qwen3-0.6b-state-lifetime-tutor-n250-v2` | 75% | 100% | 88% |
| 500 | `machalek29/qwen3-0.6b-state-lifetime-tutor-n500-v2` | 83% | 100% | 88% |
