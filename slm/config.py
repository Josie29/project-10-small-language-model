from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

MAX_TOKENS = 1024
DEFAULT_CONCURRENCY = 8

class Family(StrEnum):
    """Model family. The ablation requires at least two distinct families."""

    ANTHROPIC = "anthropic"
    MOONSHOT = "moonshot"
    OPENAI = "openai"


class Backend(StrEnum):
    """Which SDK talks to this model."""

    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"


class ModelSpec(BaseModel):
    """Everything needed to reach one model. Swap models by editing this, not the code."""

    model_id: str
    family: Family
    backend: Backend
    api_key_env: str
    base_url: str | None = None
    # Anthropic only. On pre-4.6 models (Haiku 4.5) omitting `thinking` already means
    # no thinking, and an explicit "disabled" is not an accepted value. On 4.6+ models
    # omitting it can leave adaptive thinking ON, which would contaminate the strategy
    # variable — set this True there.
    send_thinking_disabled: bool = False
    # OpenRouter only. Passed through as `extra_body`. Two uses:
    #   provider  - pin which upstream serves the request. Open-weight models route
    #               across many providers at differing quantizations (int4..bf16), so
    #               an unpinned run does not reproduce. First-party models (Anthropic,
    #               OpenAI) have a single upstream and need no pin.
    #   reasoning - disable thinking on reasoning models. Left on, they spend the whole
    #               token budget on reasoning and return empty content, and thinking
    #               would contaminate the strategy variable besides.
    extra_body: dict[str, object] | None = None

# FROZEN, and deliberately a third family: the judge must not be one of the models it
# scores. It grades the ablation AND every later base-vs-tuned run, so changing it
# mid-project invalidates comparison across runs.
JUDGE = ModelSpec(
    model_id="openai/gpt-5.6-luna",
    family=Family.OPENAI,
    backend=Backend.OPENAI_COMPATIBLE,
    api_key_env="OPENROUTER_API_KEY",
    base_url="https://openrouter.ai/api/v1",
)


def load_env_file(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE lines from a .env file into the environment.

    Existing environment variables win, so an exported key overrides the file.
    Silently does nothing when the file is absent.

    Args:
        path: Location of the .env file.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
