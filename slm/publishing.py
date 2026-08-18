from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from slm.dataset import TRAIN_SYSTEM_PROMPT
from slm.training import TrainResult


def hub_token() -> str:
    """Return a Hugging Face token from the environment or the CLI login.

    Returns:
        The token string.

    Raises:
        RuntimeError: If neither source has one.
    """
    from huggingface_hub import get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("no Hugging Face token; run `hf auth login` or set HF_TOKEN")
    return str(token)


def push_checkpoint(result: TrainResult, private: bool = False) -> str:
    """Upload the merged model and its adapter to the Hub.

    The merged model is the artifact a grader pulls: a plain causal LM that loads with
    `AutoModelForCausalLM` and needs neither peft nor the base repo. The adapter goes to a
    sibling repo because it is what makes the run auditable - a few megabytes that show
    exactly which weights moved.

    Args:
        result: A finished run, carrying the local weight directories.
        private: Whether the repos start private.

    Returns:
        The merged repo's commit sha - the thing a submission pins against.

    Raises:
        RuntimeError: If no Hugging Face token is available.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=hub_token())
    api.create_repo(result.repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=result.repo_id,
        folder_path=result.merged_dir,
        commit_message=f"Merged 16-bit checkpoint, n={result.dataset_size}",
    )
    adapter_repo = f"{result.repo_id}-adapter"
    api.create_repo(adapter_repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=adapter_repo,
        folder_path=result.adapter_dir,
        commit_message=f"LoRA adapter, n={result.dataset_size}",
    )
    return str(api.model_info(result.repo_id).sha)


def render_model_card(
    result: TrainResult,
    base_model: str,
    dataset_repo: str,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Render the model card for one checkpoint.

    Leads with the system prompt because this model is useless without it: it was trained
    against one specific line and does not carry the behavior spec in its prompt.

    Args:
        result: The finished run.
        base_model: The checkpoint's starting point.
        dataset_repo: Hub id of the published training set.
        metrics: Optional eval numbers, keyed by metric name.

    Returns:
        A Markdown model card with YAML frontmatter.
    """
    quantized = result.versions.get("quantized_base", "unknown")
    lines = [
        "---",
        f"base_model: {base_model}",
        f"datasets:\n  - {dataset_repo}",
        "library_name: transformers",
        "license: apache-2.0",
        "language:\n  - en",
        "tags:\n  - lora\n  - sft\n  - python\n  - tutoring",
        "---",
        "",
        f"# Python State-Lifetime Tutor (n={result.dataset_size})",
        "",
        "Given a short Python program with one mutable-state lifetime bug, this model quotes",
        "or identifies the relevant declaration, assignment, or mutation and asks **exactly one**",
        "non-compound question about when the object is created, who owns it, or which references",
        "share it. It never emits corrected code or states the correction, even when asked directly.",
        "",
        "## Use it with this system prompt",
        "",
        "The behavior lives in the weights, not the prompt. Send this line verbatim - the model",
        "was trained against it and nothing else:",
        "",
        # Four backticks: the snippet itself contains a fenced block, and a three-backtick
        # outer fence would be closed by it and wreck the rest of the card.
        "````python",
        "fence = chr(96) * 3",
        'user = f"{fence}python\\n{code}\\n{fence}\\n{student_message}"',
        "messages = [",
        f'    {{"role": "system", "content": "{TRAIN_SYSTEM_PROMPT}"}},',
        '    {"role": "user", "content": user},',
        "]",
        "text = tokenizer.apply_chat_template(",
        "    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False",
        ")",
        "````",
        "",
        "Thinking must be **off** and decoding greedy (`do_sample=False`); that is how it was",
        "trained and how every reported number was measured.",
        "",
        "## Training",
        "",
        "| | |",
        "|---|---|",
        f"| Base model | `{base_model}` |",
        f"| Dataset | [`{dataset_repo}`](https://huggingface.co/datasets/{dataset_repo}), first {result.dataset_size} examples by rank |",
        f"| Method | LoRA r=16, alpha=16, all linear projections, loss on the reply only |",
        f"| Frozen base precision | `{quantized}` |",
        f"| Steps | {result.steps} |",
        f"| Final training loss | {result.train_loss:.4f} |",
        f"| Wall clock | {result.runtime_seconds:.0f}s |",
        "",
        "Versions: "
        + ", ".join(f"`{k}={v}`" for k, v in sorted(result.versions.items())),
        "",
    ]
    if metrics:
        lines += [
            "## Evaluation",
            "",
            "Scored by a frozen LLM judge against the behavior spec, on 36 held-out scenarios",
            "(24 clean, 12 adversarial) that never appear in training.",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
        lines += [f"| {name} | {value} |" for name, value in metrics.items()]
        lines.append("")
    return "\n".join(lines)


def upload_card(repo_id: str, card: str, repo_type: str = "model") -> str:
    """Upload a README to a Hub repo.

    Args:
        repo_id: Target repo.
        card: Markdown content.
        repo_type: "model", "dataset", or "space".

    Returns:
        The repo's commit sha after the upload. A card push moves HEAD past whatever the
        weights landed on, so the manifest has to be refreshed or it pins a commit that no
        longer describes the repo a grader will clone.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=hub_token())
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="Update card",
    )
    return str(api.repo_info(repo_id, repo_type=repo_type).sha)


def upload_directory(
    repo_id: str, folder: Path, repo_type: str, private: bool = False
) -> str:
    """Create a repo if needed and upload a directory into it.

    Args:
        repo_id: Target repo.
        folder: Local directory to upload.
        repo_type: "model", "dataset", or "space".
        private: Whether a newly created repo starts private.

    Returns:
        The repo's commit sha after the upload.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=hub_token())
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(folder),
        commit_message=f"Upload {folder.name}",
    )
    info = api.repo_info(repo_id, repo_type=repo_type)
    return str(info.sha)
