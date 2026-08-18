from __future__ import annotations

import asyncio
import importlib
import re
from collections.abc import Sequence
from typing import Any

from slm.config import MAX_TOKENS, Backend, Family, ModelSpec
from slm.providers import Turn

# Qwen3 emits a reasoning block when its chat template is rendered with thinking enabled.
# Every call site here disables it, but a tuned checkpoint can still learn to open one, and
# a stray block would inflate `question_count` and trip `emitted_code` in the mechanical
# check. Strip it before anything scores the text.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_CLOSE = "</think>"
_THINK_OPEN = "<think>"


def strip_thinking(text: str) -> str:
    """Remove Qwen-style reasoning blocks from a completion.

    Handles three shapes: a complete block, an opener the model never closed because
    generation hit the token cap, and a closer that arrived without its opener because the
    template pre-filled one into the prompt.

    Args:
        text: Raw decoded completion.

    Returns:
        The response text with any reasoning removed.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    opener = cleaned.find(_THINK_OPEN)
    if opener != -1:
        cleaned = cleaned[:opener]
    closer = cleaned.rfind(_THINK_CLOSE)
    if closer != -1:
        cleaned = cleaned[closer + len(_THINK_CLOSE) :]
    return cleaned.strip()


def resolve_device(requested: str | None = None) -> str:
    """Pick the fastest available torch device.

    Args:
        requested: Explicit device string, or None to autodetect.

    Returns:
        One of "cuda", "mps", or "cpu".

    Raises:
        ImportError: If torch is not installed. Install the `local` extra.
    """
    if requested is not None:
        return requested
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "the local backend needs torch: uv pip install -e '.[local]'"
        ) from exc
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TransformersProvider:
    """A Hugging Face checkpoint run in this process.

    Exists so `eval.py` needs no serving account: at 0.6B a 36-scenario pass takes a few
    minutes on Apple Silicon, which is what makes the brief's "graders pull and run it
    themselves" requirement true without handing anyone a GPU bill.

    Weights load in `__init__`, so construct one only when a run is actually going ahead.
    """

    def __init__(
        self, spec: ModelSpec, device: str | None = None, revision: str | None = None
    ) -> None:
        """Load a checkpoint and its tokenizer.

        Args:
            spec: The model to run. Only `model_id`, `family`, and `temperature` are used;
                there is no endpoint or API key involved.
            device: Torch device, or None to autodetect.
            revision: Hub revision to pin. Pass the commit sha for a reproducible run.

        Raises:
            ImportError: If torch or transformers is missing. Install the `local` extra.
        """
        try:
            import torch

            # Imported as a module rather than by symbol so its partial type information
            # stays behind an explicit Any. See the same note in slm/sft.py.
            transformers: Any = importlib.import_module("transformers")
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "the local backend needs torch and transformers: "
                "uv pip install -e '.[train]'"
            ) from exc

        self.model_id = spec.model_id
        self.family = spec.family
        self._temperature = spec.temperature
        self.device = resolve_device(device)
        # float32 on CPU: bfloat16 matmuls fall back to slow kernels there, and the
        # accuracy headroom is irrelevant for a 36-prompt greedy pass.
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16

        tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
            spec.model_id, revision=revision
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer: Any = tokenizer
        model: Any = transformers.AutoModelForCausalLM.from_pretrained(
            spec.model_id, revision=revision, dtype=dtype
        )
        self._model: Any = model.to(self.device)
        self._model.eval()
        # One model, one set of weights: overlapping generate() calls from the harness's
        # worker threads would contend rather than parallelise. Serialise instead, and let
        # the judge calls be what actually overlaps.
        self._lock = asyncio.Lock()

    def _generate(self, system: str, turns: Sequence[Turn]) -> str:
        """Render the chat template and greedily decode one completion.

        Args:
            system: System prompt.
            turns: Conversation turns ending with the student's message.

        Returns:
            The decoded completion with any reasoning block stripped.
        """
        import torch

        messages = [{"role": "system", "content": system}]
        messages += [{"role": t.role.value, "content": t.content} for t in turns]
        # enable_thinking=False must match what training used. With the flag set, Qwen3's
        # template closes an empty reasoning block for you; if train and inference disagree
        # here the tuned model's output format drifts from what the judge was shown.
        text: str = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        sampling = (
            {"do_sample": False}
            if self._temperature == 0.0
            else {"do_sample": True, "temperature": self._temperature}
        )
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                pad_token_id=self._tokenizer.pad_token_id,
                **sampling,
            )
        prompt_length: int = inputs["input_ids"].shape[-1]
        completion: str = self._tokenizer.decode(
            output[0][prompt_length:], skip_special_tokens=True
        )
        return strip_thinking(completion)

    async def complete(self, system: str, turns: Sequence[Turn]) -> str:
        """Return the checkpoint's response text.

        Args:
            system: System prompt for this model's role.
            turns: Conversation turns, ending with the student's message.

        Returns:
            The response text, or an empty string if the model produced none.
        """
        async with self._lock:
            return await asyncio.to_thread(self._generate, system, turns)


def local_spec(model_id: str) -> ModelSpec:
    """Build a spec for a checkpoint run in-process.

    Args:
        model_id: Hugging Face repo id or local path.

    Returns:
        A spec carrying no endpoint or key, since nothing is called over the network.
    """
    return ModelSpec(
        model_id=model_id,
        family=Family.QWEN,
        backend=Backend.TRANSFORMERS,
        # Never read: TransformersProvider makes no network call. ModelSpec requires the
        # field because every other backend does.
        api_key_env="",
    )
