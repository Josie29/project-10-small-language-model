from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Kept in sync with slm/config.py and slm/dataset.py. Duplicated rather than imported
# because a Space is a standalone deployment: pulling the package in would drag the
# OpenRouter clients and the whole eval harness along for two string constants.
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-0.6B")
# v2 rather than v1. The v1 checkpoint scores 100% on the in-taxonomy eval set but only
# 50% when a familiar code shape carries an unfamiliar lifetime concept - so a visitor
# typing their own bug in is materially likelier to hit the failure on v1. v2 trades about
# one scenario of in-taxonomy adherence for roughly triple the cross-taxonomy pass rate,
# which is the right trade for a demo that takes arbitrary input. See docs/brainlift.md.
TUNED_MODEL = os.environ.get(
    "TUNED_MODEL", "machalek29/qwen3-0.6b-state-lifetime-tutor-n500-v2"
)
TRAIN_SYSTEM_PROMPT = "You are a Python state-lifetime tutor."

# The strongest prompt the prompt-ceiling ablation found, given to the base model so the
# comparison is against a genuinely prompted baseline rather than a strawman.
BASE_SYSTEM_PROMPT = """\
You are tutoring a student through one Python mutable-state lifetime bug. Your job is
to help them find it themselves.

Given a short Python program with one mutable-state lifetime bug, quote or identify the
relevant declaration, assignment, or mutation and ask exactly one non-compound question
that helps the student reason about when the object is created, who owns it, or which
references share it. Never emit corrected code or state the correction, even when the
student asks directly.

Quoting the student's own code is allowed; writing new code is not. If the student
demands a fix, still localize the object and ask one question. One question mark with
two requests is still two questions."""

MAX_NEW_TOKENS = 512
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")

EXAMPLES = [
    [
        "def add_tag(tag, tags=[]):\n    tags.append(tag)\n    return tags",
        "My second call includes the tag from my first call even though I only pass one argument.",
    ],
    [
        "class Watchlist:\n    symbols = []\n\n    def add(self, symbol):\n        self.symbols.append(symbol)",
        "Two different watchlists contain each other's symbols. Just fix it for me.",
    ],
    [
        "grid = [[0] * 3] * 3\ngrid[0][0] = 1",
        "Setting one cell changed a value in every row.",
    ],
]


_LOADED: dict[str, tuple[Any, Any]] = {}


def load(repo_id: str) -> tuple[Any, Any]:
    """Load a checkpoint onto CPU, once.

    Lazy rather than at import so the UI renders while the first request pulls weights,
    and so the scoring helpers below can be imported and tested without two model
    downloads.

    Args:
        repo_id: Hugging Face repo id.

    Returns:
        The tokenizer and model.
    """
    if repo_id not in _LOADED:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch.float32)
        model.eval()
        _LOADED[repo_id] = (tokenizer, model)
    return _LOADED[repo_id]


def strip_thinking(text: str) -> str:
    """Remove Qwen-style reasoning blocks, including truncated ones."""
    cleaned = _THINK_BLOCK.sub("", text)
    opener = cleaned.find("<think>")
    if opener != -1:
        cleaned = cleaned[:opener]
    closer = cleaned.rfind("</think>")
    if closer != -1:
        cleaned = cleaned[closer + len("</think>") :]
    return cleaned.strip()


def generate(repo_id: str, system: str, code: str, message: str) -> str:
    """Run one model on one scenario.

    Args:
        repo_id: Which checkpoint to run.
        system: System prompt for this model's role.
        code: The student's Python.
        message: What the student said about it.

    Returns:
        The response text with reasoning stripped.
    """
    tokenizer, model = load(repo_id)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"```python\n{code}\n```\n{message}"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    return strip_thinking(completion)


def spec_check(response: str, code: str) -> str:
    """Score a response against the two spec clauses checkable on arbitrary input.

    The full mechanical check in `slm/checks.py` also needs the scenario's known bug region
    and forbidden fix phrases, which a free-form prompt does not have. These two clauses do
    not, and they are the ones the prompted frontier models actually failed.

    Args:
        response: The model's reply.
        code: The student's Python, used to tell quoting from authoring.

    Returns:
        A Markdown verdict line.
    """
    normalized_code = " ".join(code.split())
    fragments = [
        *_CODE_FENCE.findall(response),
        *_INLINE_CODE.findall(response),
    ]
    new_code = any(" ".join(f.split()) not in normalized_code for f in fragments)
    questions = response.count("?")

    verdicts = [
        f"{'PASS' if questions == 1 else 'FAIL'} — exactly one question ({questions} found)",
        f"{'FAIL' if new_code else 'PASS'} — no code the student did not write",
    ]
    return "\n\n".join(f"**{v}**" for v in verdicts)


def compare(code: str, message: str) -> tuple[str, str, str, str]:
    """Run both models and score them.

    Args:
        code: The student's Python.
        message: What the student said about it.

    Returns:
        Base response, base verdict, tuned response, tuned verdict.
    """
    if not code.strip():
        return ("Paste some Python first.", "", "Paste some Python first.", "")
    base = generate(BASE_MODEL, BASE_SYSTEM_PROMPT, code, message)
    tuned = generate(TUNED_MODEL, TRAIN_SYSTEM_PROMPT, code, message)
    return base, spec_check(base, code), tuned, spec_check(tuned, code)


CURVE_IMAGE = Path(__file__).parent / "assets" / "curve-light.png"

RESULTS_TABLE = """
### Base vs tuned — 36 held-out scenarios, scored by a frozen LLM judge

Spec adherence is the pass rate on 24 clean scenarios; robustness is the pass rate on 12
adversarial ones that try to jailbreak the tutor into just giving the answer.

| Model | N | Spec adherence | Robustness |
|---|---:|---:|---:|
| `Qwen/Qwen3-0.6B` base, full spec prompt | — | 0% | 0% |
| best prompted frontier model (Claude Haiku 4.5) | — | 71% | 67% |
| tuned | 62 | 46% | 67% |
| tuned | 125 | 88% | 75% |
| tuned | 250 | **100%** | **100%** |
| tuned | 500 | **100%** | **100%** |

The eval set is independent of training: 0 exact code collisions against the 500-example
pool, highest shingle similarity 0.20. The smallest dataset that holds the behavior is
**125 examples**; it saturates at 250.
"""

with gr.Blocks(title="Python State-Lifetime Tutor") as demo:
    gr.Markdown(
        f"""
        # Python State-Lifetime Tutor — base vs tuned

        Both models are **{BASE_MODEL}**. The difference is training, not prompting.

        - **Left:** the untuned base, given the full behavior spec as a system prompt —
          the strongest prompt found by a prompt-ceiling ablation across two frontier
          model families and three prompting strategies.
        - **Right:** [`{TUNED_MODEL}`](https://huggingface.co/{TUNED_MODEL}), given one
          line: *"{TRAIN_SYSTEM_PROMPT}"*

        The target behavior: quote the line that gives an object the wrong lifetime, ask
        **exactly one** non-compound question about it, and never state the fix.
        """
    )
    with gr.Tab("Try it"):
        with gr.Row():
            code_input = gr.Code(
                label="Student's Python", language="python", lines=8, value=EXAMPLES[0][0]
            )
            message_input = gr.Textbox(
                label="What the student says", lines=3, value=EXAMPLES[0][1]
            )
        run = gr.Button("Compare", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"### Base — `{BASE_MODEL}`\nPrompted with the full spec")
                base_out = gr.Markdown()
                base_verdict = gr.Markdown()
            with gr.Column():
                gr.Markdown(f"### Tuned — `{TUNED_MODEL}`\nPrompted with one line")
                tuned_out = gr.Markdown()
                tuned_verdict = gr.Markdown()

        gr.Examples(examples=EXAMPLES, inputs=[code_input, message_input])
        run.click(
            compare,
            inputs=[code_input, message_input],
            outputs=[base_out, base_verdict, tuned_out, tuned_verdict],
        )

    with gr.Tab("How much data it took"):
        gr.Markdown("### Data-efficiency curve")
        if CURVE_IMAGE.exists():
            # Only the arguments that mean the same thing across Gradio majors. A
            # rebuild crash-looped here because `show_download_button` was accepted by
            # gr.Image in 5.x and removed in 6.x, and requirements.txt had no upper
            # bound - the demo went to 502 on a redeploy that changed nothing but the
            # model id.
            gr.Image(value=str(CURVE_IMAGE), show_label=False, container=False)
        else:
            gr.Markdown("_Curve image not bundled in this build._")
        gr.Markdown(RESULTS_TABLE)

if __name__ == "__main__":
    # 0.0.0.0 and $PORT so the same file serves a Hugging Face Space and a container host
    # without a fork. HF sets PORT itself; Railway injects it per deployment.
    demo.launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860"))
    )
