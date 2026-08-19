# Base vs Tuned — Results

Spec adherence = judge pass rate on clean scenarios.
Robustness = judge pass rate on adversarial scenarios.

| Model | Strategy | Spec adherence | Robustness | Mechanical pass | n |
| --- | --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-0.6B` | zero_shot | 0% | 0% | 0% | 36 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n62` | zero_shot | 38% | 67% | 83% | 36 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n125` | zero_shot | 79% | 67% | 100% | 36 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n250` | zero_shot | 100% | 100% | 100% | 36 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n500` | zero_shot | 100% | 100% | 97% | 36 |

## Judge pass rate by state/lifetime concept

| Model | Strategy | Concept | Pass rate | n |
| --- | --- | --- | ---: | ---: |
| `Qwen/Qwen3-0.6B` | zero_shot | creation | 0% | 9 |
| `Qwen/Qwen3-0.6B` | zero_shot | ownership | 0% | 9 |
| `Qwen/Qwen3-0.6B` | zero_shot | reset | 0% | 9 |
| `Qwen/Qwen3-0.6B` | zero_shot | aliasing | 0% | 9 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n62` | zero_shot | creation | 67% | 9 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n62` | zero_shot | ownership | 56% | 9 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n62` | zero_shot | reset | 67% | 9 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n62` | zero_shot | aliasing | 0% | 9 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n125` | zero_shot | creation | 100% | 9 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n125` | zero_shot | ownership | 67% | 9 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n125` | zero_shot | reset | 67% | 9 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n125` | zero_shot | aliasing | 67% | 9 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n250` | zero_shot | creation | 100% | 9 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n250` | zero_shot | ownership | 100% | 9 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n250` | zero_shot | reset | 100% | 9 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n250` | zero_shot | aliasing | 100% | 9 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n500` | zero_shot | creation | 100% | 9 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n500` | zero_shot | ownership | 100% | 9 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n500` | zero_shot | reset | 100% | 9 |
| `machalek29/qwen3-0.6b-state-lifetime-tutor-n500` | zero_shot | aliasing | 100% | 9 |

## Violations across all cells

| Violation | Count |
| --- | ---: |
| wrong_lifetime_focus | 42 |
| no_question | 9 |
| stated_fix | 5 |
| emitted_code | 4 |
| multiple_questions | 3 |
| no_localization | 1 |
