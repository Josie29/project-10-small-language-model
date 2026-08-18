# Data-Efficiency Curve

Spec adherence = judge pass rate on clean scenarios.
Robustness = judge pass rate on adversarial scenarios.

Epochs are held fixed across every point, so the smallest N also trains for the
fewest optimizer steps. Read the low end as data *and* step starvation together.

| N | Checkpoint | Spec adherence | Robustness | Mechanical pass |
| ---: | --- | ---: | ---: | ---: |
| 62 | `machalek29/qwen3-0.6b-state-lifetime-tutor-n62` | 46% | 67% | 83% |
| 125 | `machalek29/qwen3-0.6b-state-lifetime-tutor-n125` | 88% | 75% | 100% |
| 250 | `machalek29/qwen3-0.6b-state-lifetime-tutor-n250` | 100% | 100% | 100% |
| 500 | `machalek29/qwen3-0.6b-state-lifetime-tutor-n500` | 100% | 100% | 97% |
