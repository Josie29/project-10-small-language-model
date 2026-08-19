# Base vs Tuned — Results

Spec adherence = judge pass rate on clean scenarios.
Robustness = judge pass rate on adversarial scenarios.

| Model | Strategy | Spec adherence | Robustness | Mechanical pass | n |
| --- | --- | ---: | ---: | ---: | ---: |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n62-v2` | zero_shot | 58% | 50% | 69% | 16 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n125-v2` | zero_shot | 58% | 75% | 88% | 16 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n250-v2` | zero_shot | 75% | 100% | 88% | 16 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n500-v2` | zero_shot | 83% | 100% | 88% | 16 |

## Judge pass rate by state/lifetime concept

| Model | Strategy | Concept | Pass rate | n |
| --- | --- | --- | ---: | ---: |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n62-v2` | zero_shot | creation | 100% | 2 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n62-v2` | zero_shot | ownership | 75% | 4 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n62-v2` | zero_shot | reset | 60% | 5 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n62-v2` | zero_shot | aliasing | 20% | 5 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n125-v2` | zero_shot | creation | 100% | 2 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n125-v2` | zero_shot | ownership | 25% | 4 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n125-v2` | zero_shot | reset | 100% | 5 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n125-v2` | zero_shot | aliasing | 40% | 5 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n250-v2` | zero_shot | creation | 100% | 2 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n250-v2` | zero_shot | ownership | 75% | 4 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n250-v2` | zero_shot | reset | 80% | 5 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n250-v2` | zero_shot | aliasing | 80% | 5 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n500-v2` | zero_shot | creation | 100% | 2 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n500-v2` | zero_shot | ownership | 75% | 4 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n500-v2` | zero_shot | reset | 100% | 5 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n500-v2` | zero_shot | aliasing | 80% | 5 |

## Violations across all cells

| Violation | Count |
| --- | ---: |
| wrong_lifetime_focus | 18 |
