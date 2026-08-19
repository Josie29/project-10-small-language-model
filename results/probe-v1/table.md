# Base vs Tuned — Results

Spec adherence = judge pass rate on clean scenarios.
Robustness = judge pass rate on adversarial scenarios.

| Model | Strategy | Spec adherence | Robustness | Mechanical pass | n |
| --- | --- | ---: | ---: | ---: | ---: |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n62` | zero_shot | 58% | 50% | 69% | 16 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n125` | zero_shot | 42% | 50% | 75% | 16 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n250` | zero_shot | 58% | 25% | 75% | 16 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n500` | zero_shot | 83% | 25% | 62% | 16 |

## Judge pass rate by state/lifetime concept

| Model | Strategy | Concept | Pass rate | n |
| --- | --- | --- | ---: | ---: |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n62` | zero_shot | creation | 100% | 2 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n62` | zero_shot | ownership | 50% | 4 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n62` | zero_shot | reset | 80% | 5 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n62` | zero_shot | aliasing | 20% | 5 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n125` | zero_shot | creation | 100% | 2 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n125` | zero_shot | ownership | 50% | 4 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n125` | zero_shot | reset | 20% | 5 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n125` | zero_shot | aliasing | 40% | 5 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n250` | zero_shot | creation | 100% | 2 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n250` | zero_shot | ownership | 75% | 4 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n250` | zero_shot | reset | 20% | 5 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n250` | zero_shot | aliasing | 40% | 5 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n500` | zero_shot | creation | 100% | 2 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n500` | zero_shot | ownership | 100% | 4 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n500` | zero_shot | reset | 60% | 5 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n500` | zero_shot | aliasing | 40% | 5 |

## Violations across all cells

| Violation | Count |
| --- | ---: |
| wrong_lifetime_focus | 25 |
| no_localization | 4 |
