# Prompt-Ceiling Ablation — Results

Spec adherence = judge pass rate on clean scenarios.
Robustness = judge pass rate on adversarial scenarios.

| Model | Strategy | Spec adherence | Robustness | Mechanical pass | n |
| --- | --- | ---: | ---: | ---: | ---: |
| `anthropic/claude-haiku-4.5` | zero_shot | 71% | 67% | 11% | 36 |
| `anthropic/claude-haiku-4.5` | few_shot | 54% | 50% | 25% | 36 |
| `anthropic/claude-haiku-4.5` | structured_cot | 62% | 58% | 33% | 36 |
| `moonshotai/kimi-k2.6` | zero_shot | 58% | 42% | 19% | 36 |
| `moonshotai/kimi-k2.6` | few_shot | 67% | 58% | 47% | 36 |
| `moonshotai/kimi-k2.6` | structured_cot | 46% | 25% | 22% | 36 |

## Judge pass rate by state/lifetime concept

| Model | Strategy | Concept | Pass rate | n |
| --- | --- | --- | ---: | ---: |
| `anthropic/claude-haiku-4.5` | zero_shot | creation | 89% | 9 |
| `anthropic/claude-haiku-4.5` | zero_shot | ownership | 56% | 9 |
| `anthropic/claude-haiku-4.5` | zero_shot | reset | 56% | 9 |
| `anthropic/claude-haiku-4.5` | zero_shot | aliasing | 78% | 9 |
| `anthropic/claude-haiku-4.5` | few_shot | creation | 22% | 9 |
| `anthropic/claude-haiku-4.5` | few_shot | ownership | 22% | 9 |
| `anthropic/claude-haiku-4.5` | few_shot | reset | 67% | 9 |
| `anthropic/claude-haiku-4.5` | few_shot | aliasing | 100% | 9 |
| `anthropic/claude-haiku-4.5` | structured_cot | creation | 67% | 9 |
| `anthropic/claude-haiku-4.5` | structured_cot | ownership | 11% | 9 |
| `anthropic/claude-haiku-4.5` | structured_cot | reset | 78% | 9 |
| `anthropic/claude-haiku-4.5` | structured_cot | aliasing | 89% | 9 |
| `moonshotai/kimi-k2.6` | zero_shot | creation | 33% | 9 |
| `moonshotai/kimi-k2.6` | zero_shot | ownership | 22% | 9 |
| `moonshotai/kimi-k2.6` | zero_shot | reset | 100% | 9 |
| `moonshotai/kimi-k2.6` | zero_shot | aliasing | 56% | 9 |
| `moonshotai/kimi-k2.6` | few_shot | creation | 89% | 9 |
| `moonshotai/kimi-k2.6` | few_shot | ownership | 0% | 9 |
| `moonshotai/kimi-k2.6` | few_shot | reset | 89% | 9 |
| `moonshotai/kimi-k2.6` | few_shot | aliasing | 78% | 9 |
| `moonshotai/kimi-k2.6` | structured_cot | creation | 0% | 9 |
| `moonshotai/kimi-k2.6` | structured_cot | ownership | 0% | 9 |
| `moonshotai/kimi-k2.6` | structured_cot | reset | 89% | 9 |
| `moonshotai/kimi-k2.6` | structured_cot | aliasing | 67% | 9 |

## Violations across all cells

| Violation | Count |
| --- | ---: |
| multiple_questions | 77 |
| stated_fix | 10 |
| emitted_code | 3 |
| wrong_lifetime_focus | 3 |
| no_question | 1 |
