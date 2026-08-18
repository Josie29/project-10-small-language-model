from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

from slm.training import TrainConfig, TrainResult

# The CUDA half of the training path. Split out of `train.py` so importing the CLI does not
# require Modal to be installed, and so the local backend stays usable when GPU billing is
# not available.
#
# Versions are deliberately unpinned on the first build: a guessed pin is likelier to break
# the build than to help. The resolved set comes back in TrainResult.versions, which is what
# a later run should pin against.

app = modal.App("slm-state-lifetime-train")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "trl",
        "peft",
        "accelerate",
        "datasets",
        "bitsandbytes",
        "huggingface_hub",
        "hf_transfer",
        # `slm` is added below and its package __init__ imports these. Pure-python wheels.
        "pydantic",
        "openai",
        "anthropic",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf"})
    .add_local_python_source("slm")
)

# Survives between runs so a four-point sweep downloads the base model once.
hf_cache = modal.Volume.from_name("slm-hf-cache", create_if_missing=True)


@app.function(  # pyright: ignore[reportUnknownMemberType] - Modal's decorator factory is only partially typed
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/cache": hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
)
def train_one(config: TrainConfig, rows: list[dict[str, Any]]) -> TrainResult:
    """Fine-tune one curve point on a GPU with a true 4-bit NF4 base.

    Args:
        config: Hyperparameters and destination repo.
        rows: Chat-format training rows.

    Returns:
        The finished run, with `revision` set - the weights only ever exist inside this
        container, so the push has to happen here rather than at the call site.
    """
    from slm.publishing import push_checkpoint
    from slm.sft import run_sft

    result = run_sft(
        config,
        rows,
        output_dir=Path("/tmp/train") / f"n-{config.dataset_size}",
        device="cuda",
        load_in_4bit=True,
    )
    result.revision = push_checkpoint(result)
    return result
