# Base vs Tuned — Results

Spec adherence = judge pass rate on clean scenarios.
Robustness = judge pass rate on adversarial scenarios.

| Model | Strategy | Spec adherence | Robustness | Mechanical pass | n |
| --- | --- | ---: | ---: | ---: | ---: |
| `checkpoints/local-eval/tutor-n62-v2` | zero_shot | 50% | 50% | 69% | 16 |
| `checkpoints/local-eval/tutor-n125-v2` | zero_shot | 64% | 75% | 93% | 15 |
| `checkpoints/local-eval/tutor-n250-v2` | zero_shot | 83% | 100% | 88% | 16 |
| `checkpoints/local-eval/tutor-n500-v2` | zero_shot | 83% | 100% | 88% | 16 |

## Judge pass rate by state/lifetime concept

| Model | Strategy | Concept | Pass rate | n |
| --- | --- | --- | ---: | ---: |
| `checkpoints/local-eval/tutor-n62-v2` | zero_shot | creation | 100% | 2 |
| `checkpoints/local-eval/tutor-n62-v2` | zero_shot | ownership | 50% | 4 |
| `checkpoints/local-eval/tutor-n62-v2` | zero_shot | reset | 60% | 5 |
| `checkpoints/local-eval/tutor-n62-v2` | zero_shot | aliasing | 20% | 5 |
| `checkpoints/local-eval/tutor-n125-v2` | zero_shot | creation | 100% | 2 |
| `checkpoints/local-eval/tutor-n125-v2` | zero_shot | ownership | 50% | 4 |
| `checkpoints/local-eval/tutor-n125-v2` | zero_shot | reset | 80% | 5 |
| `checkpoints/local-eval/tutor-n125-v2` | zero_shot | aliasing | 50% | 4 |
| `checkpoints/local-eval/tutor-n250-v2` | zero_shot | creation | 100% | 2 |
| `checkpoints/local-eval/tutor-n250-v2` | zero_shot | ownership | 75% | 4 |
| `checkpoints/local-eval/tutor-n250-v2` | zero_shot | reset | 80% | 5 |
| `checkpoints/local-eval/tutor-n250-v2` | zero_shot | aliasing | 100% | 5 |
| `checkpoints/local-eval/tutor-n500-v2` | zero_shot | creation | 100% | 2 |
| `checkpoints/local-eval/tutor-n500-v2` | zero_shot | ownership | 75% | 4 |
| `checkpoints/local-eval/tutor-n500-v2` | zero_shot | reset | 100% | 5 |
| `checkpoints/local-eval/tutor-n500-v2` | zero_shot | aliasing | 80% | 5 |

## Violations across all cells

| Violation | Count |
| --- | ---: |
| wrong_lifetime_focus | 16 |
| no_localization | 1 |
