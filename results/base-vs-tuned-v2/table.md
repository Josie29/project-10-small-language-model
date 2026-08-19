# Base vs Tuned — Results

Spec adherence = judge pass rate on clean scenarios.
Robustness = judge pass rate on adversarial scenarios.

| Model | Strategy | Spec adherence | Robustness | Mechanical pass | n |
| --- | --- | ---: | ---: | ---: | ---: |
| `checkpoints/local-eval/tutor-n62-v2` | zero_shot | 50% | 67% | 86% | 36 |
| `checkpoints/local-eval/tutor-n125-v2` | zero_shot | 88% | 92% | 97% | 36 |
| `checkpoints/local-eval/tutor-n250-v2` | zero_shot | 100% | 100% | 100% | 36 |
| `checkpoints/local-eval/tutor-n500-v2` | zero_shot | 96% | 100% | 100% | 36 |

## Judge pass rate by state/lifetime concept

| Model | Strategy | Concept | Pass rate | n |
| --- | --- | --- | ---: | ---: |
| `checkpoints/local-eval/tutor-n62-v2` | zero_shot | creation | 78% | 9 |
| `checkpoints/local-eval/tutor-n62-v2` | zero_shot | ownership | 67% | 9 |
| `checkpoints/local-eval/tutor-n62-v2` | zero_shot | reset | 67% | 9 |
| `checkpoints/local-eval/tutor-n62-v2` | zero_shot | aliasing | 11% | 9 |
| `checkpoints/local-eval/tutor-n125-v2` | zero_shot | creation | 100% | 9 |
| `checkpoints/local-eval/tutor-n125-v2` | zero_shot | ownership | 67% | 9 |
| `checkpoints/local-eval/tutor-n125-v2` | zero_shot | reset | 100% | 9 |
| `checkpoints/local-eval/tutor-n125-v2` | zero_shot | aliasing | 89% | 9 |
| `checkpoints/local-eval/tutor-n250-v2` | zero_shot | creation | 100% | 9 |
| `checkpoints/local-eval/tutor-n250-v2` | zero_shot | ownership | 100% | 9 |
| `checkpoints/local-eval/tutor-n250-v2` | zero_shot | reset | 100% | 9 |
| `checkpoints/local-eval/tutor-n250-v2` | zero_shot | aliasing | 100% | 9 |
| `checkpoints/local-eval/tutor-n500-v2` | zero_shot | creation | 89% | 9 |
| `checkpoints/local-eval/tutor-n500-v2` | zero_shot | ownership | 100% | 9 |
| `checkpoints/local-eval/tutor-n500-v2` | zero_shot | reset | 100% | 9 |
| `checkpoints/local-eval/tutor-n500-v2` | zero_shot | aliasing | 100% | 9 |

## Violations across all cells

| Violation | Count |
| --- | ---: |
| wrong_lifetime_focus | 19 |
| emitted_code | 2 |
