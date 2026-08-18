# Tech Stack — Python State Lifetime Tutor (Project 10)

Scope: everything downstream of the prompt-ceiling ablation, which is already built and
run (see [behavior-spec.md](behavior-spec.md), `ablation.py`, `results/state-lifetime-v1/`).
Choices below are constrained by three facts: the dev machine is an Apple Silicon Mac (no
CUDA), the judge is frozen, and a grader must be able to re-run everything from a public
checkpoint with one command.

| Layer | Component | Choice | Reason |
|---|---|---|---|
| Model | Base model | `Qwen/Qwen3-0.6B` | Target behavior is output-format discipline over four canonical bug shapes, which is where SFT on a tiny model works best — and a near-floor base makes the base-vs-tuned delta legible. |
| Model | Escalation path | `Qwen/Qwen3-1.7B`, same dataset and harness | Held in reserve behind an explicit tripwire (below) so a capacity failure costs one training run, not a rewrite. |
| Model | Decode config | Thinking off, temperature 0 | Matches the ablation's frozen sampling settings, so base-vs-tuned is a clean comparison. |
| Training | Fine-tune method | LoRA r=16 on a bf16 base — see the QLoRA deviation below | 4-bit NF4 needs bitsandbytes, which has no Metal backend. `slm/sft.py` takes `load_in_4bit` as a parameter, so the identical recipe runs as true QLoRA the moment a CUDA host is available. |
| Training | Framework | TRL `SFTTrainer` + PEFT | Unsloth is CUDA-only, so it cannot run on the machine that has the GPU budget. TRL's `completion_only_loss` covers the one thing Unsloth was wanted for (masking the prompt). |
| Training | Run logs | TRL `trainer_state.json` + loss CSV committed per checkpoint | Brief requires training logs a grader can inspect; files in git need no external account. |
| Compute | Training + eval host | Local Apple Silicon (M4, MPS) | Modal's spend limit blocked the account before the first image built. At 0.6B the whole 4-point sweep is ~20 minutes of MPS, so the GPU host was a convenience rather than a requirement. `train.py --backend modal` is retained and untested. |
| Data | Teacher model | Claude Opus 5 via OpenRouter | Strongest available teacher, and a different family from the frozen OpenAI judge, so the judge never grades its own distillate. |
| Data | Generation transport | Existing `slm/providers.py` + OpenRouter | Already written, already typed, one key for every family. |
| Data | Quality gate | `slm/checks.py` mechanical checks, then the frozen judge as accept/reject | The eval rubric *is* the filter — a sample that would fail eval never enters training. |
| Data | Format | Chat-messages JSONL, published as a Hugging Face dataset | Native to TRL's SFT path; the dataset is the graded artifact and must be public. |
| Eval | Harness | Extend the existing `slm/` package; add `eval.py --model <id> --eval-set <path>` | Rubric, judge, scenario schema, and table rendering already exist and must not change. |
| Eval | Tuned-model serving | In-process `transformers` (`slm/local.py`) | At 0.6B a 36-scenario pass is a few minutes on a laptop, so a grader needs no GPU, no serving account, and no endpoint that has to still be up at grading time. One ~60-line `Provider` implementation. |
| Publishing | Checkpoint hosting | Hugging Face Hub, public, merged 16-bit + adapter, pinned commit | Required by the brief's verification table. |
| Demo | Inference demo | Gradio app on a free CPU Hugging Face Space, base vs tuned side by side | A 0.6B pair runs on CPU with no GPU queue, so the live grader-supplied prompt in the demo video cannot stall. |
| Tooling | Language / runtime | Python 3.12+ | Already pinned in `pyproject.toml`. |
| Tooling | Dependency management | uv | Already the documented setup path in the README. |
| Tooling | Types / tests | pyright strict, pytest | Already configured; extend `include` to cover `eval.py` and the new data-generation module. |

## The QLoRA deviation

The brief names QLoRA. What shipped is LoRA on a bf16 base, and the reason is a hard
constraint rather than a preference: bitsandbytes has no Metal backend, and Modal's
account spend limit blocked the CUDA path before the first image finished building.

What is unchanged: rank, alpha, dropout, target modules (every linear projection),
learning rate, schedule, epochs, batch size, seed, and prompt masking. The only difference
is that the frozen base is held in bf16 instead of 4-bit NF4 — a memory optimization worth
about 900MB on a 0.6B model, on a machine with 16GB of unified memory. It changes what the
run costs, not what the adapter learns.

`slm/sft.py:run_sft` takes `load_in_4bit` as a parameter and `modal_app.py` passes it as
True, so re-running the sweep as literal QLoRA on any CUDA host is a flag, not a rewrite.
Every reported number records which one produced it in `versions.quantized_base`.

## Base-model escalation tripwire

0.6B is a deliberate bet that this behavior is format discipline rather than
comprehension. The bet is falsifiable, and the existing violation breakdown in
`slm/reporting.py` is what falsifies it. After the first real tune:

| Dominant violation | Diagnosis | Action |
|---|---|---|
| `multiple_questions`, `stated_fix`, `emitted_code` | Format discipline — data problem | Stay on 0.6B, fix the dataset. |
| `no_localization`, `wrong_lifetime_focus` | Cannot identify the culprit line — capacity problem | Switch to 1.7B. |

Hard tripwire: if tuned-0.6B clean adherence lands under ~80% *and* the failures are
comprehension-flavored, move to 1.7B rather than iterating on data. 80% is chosen
because the prompted frontier ceiling is 71% (Haiku 4.5 zero-shot) — a tuned model that
merely matches the ceiling proves nothing.

## Rejected alternatives

| Component | Option | Why not |
|---|---|---|
| Base model | Qwen3-1.7B as the starting point | A stronger base compresses the base-vs-tuned delta and multiplies every point on the data-efficiency curve; kept as the escalation target instead. |
| Base model | Qwen3-4B | Several times the cost across a 4-point curve for a behavior that is formatting discipline, not knowledge. |
| Base model | Llama 3.2 1B/3B, Gemma 3, SmolLM3 | Viable, but Qwen3 is the brief's stated default and has the best small-model instruction following right now. |
| Fine-tune framework | Unsloth | CUDA-only, so it cannot run on the machine that ended up doing the training. Retained in `modal_app.py`'s lineage only as the faster option once a GPU host is available. |
| Fine-tune framework | Axolotl | YAML-driven config is a second configuration surface to learn for no gain at this scale. |
| Compute | Modal A10G | The original choice; the workspace's default spend limit blocked it before the first image built, and at 0.6B the local machine turned out to be sufficient. `--backend modal` is still wired up. |
| Compute | RunPod | Manual pod lifecycle — easy to leave running, harder to hand a grader a reproducible command. |
| Compute | Colab | Session timeouts and an un-pinned environment make a 4-checkpoint sweep unreliable and unreproducible. |
| Compute | Local MLX | MLX's affine 4-bit is genuine on-Metal quantization, but the weights need converting back to HF safetensors and it is a second toolchain for a saving that does not matter at 0.6B. Plain MPS via `transformers` won instead — see the QLoRA deviation above. |
| Tuned-model serving | vLLM OpenAI-compatible server on Modal | One server serves one model, so a 5-checkpoint sweep needs five deployments or a parameterized endpoint with no stable URL — and it puts a live service on the reproduction path. |
| Teacher model | The judge model (`openai/gpt-5.6-luna`) | The judge would be grading its own generations — the filter stops being independent evidence. |
| Teacher model | A model under test (Haiku 4.5, Kimi k2.6) | Distilling from a model whose prompt ceiling we measured caps the student at that ceiling. |
| Data quality gate | Hand review only | Does not scale past a few hundred examples and is not re-runnable by a grader. |
| Data quality gate | Mechanical checks only | The dominant ablation failure (`multiple_questions`, 77 of 94) is exactly the case the regex heuristic only flags for review. |
| Tuned-model serving | HF Inference Endpoints | Extra paid always-on surface, and it puts a live service on the reproduction path. |
| Tuned-model serving | Ollama / llama.cpp | Local-only, so it fails the "grader pulls and runs it themselves" requirement. |
| Demo | HF Spaces ZeroGPU | Unnecessary at 0.6B, and its queue is a live failure mode during a recorded demo. |
| Demo | Modal web endpoint | Another service to keep warm; the checkpoint already lives on the Hub, so the Space is closer to the artifact. |
| Run logs | Weights & Biases | Adds an account and API key to the reproduction path for curves we can render from committed JSONL. |
| Eval harness | New rubric or judge for base-vs-tuned | Changing either invalidates comparison against the ablation numbers already on the board. |

## Open sub-decisions

- **Dataset size spacing (N, N/2, N/4, N/8).** Top of curve depends on how many samples survive the quality gate; fix once the first generation batch is filtered.
- **LoRA rank / alpha / epochs.** Start at Unsloth defaults; per the brief, treat a bad result as a data problem, not a knob to tune.
- **Adversarial split in the training data.** Whether to include jailbreak-shaped ("just fix it") examples in training or hold that shape out entirely to keep Robustness honest — decide before the v1 dataset.
- **Concept balance.** Ablation shows `ownership` collapsing (0–56% across every cell); decide whether to oversample it or keep the four concepts even.
