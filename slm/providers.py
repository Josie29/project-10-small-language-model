from __future__ import annotations

import os
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

import anthropic
import openai
from pydantic import BaseModel

from slm.config import MAX_TOKENS, Family, ModelSpec

class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"

class Turn(BaseModel):
    """One message in the conversation sent to a model under test."""

    role: Role
    content: str

class Provider(Protocol):
    """A frontier model that can be prompted with a system prompt and a turn list."""

    model_id: str
    family: Family

    async def complete(self, system: str, turns: Sequence[Turn]) -> str:
        """Return the model's response text.

        Args:
            system: System prompt for this prompting strategy.
            turns: Conversation turns, ending with the student's message.

        Returns:
            The response text, or an empty string if the model produced none.
        """
        ...


class AnthropicProvider:
    """Claude, via the official Anthropic SDK."""

    def __init__(self, spec: ModelSpec, client: anthropic.AsyncAnthropic) -> None:
        self.model_id = spec.model_id
        self.family = spec.family
        self._send_thinking_disabled = spec.send_thinking_disabled
        self._client = client

    async def complete(self, system: str, turns: Sequence[Turn]) -> str:
        """Send one request to Claude with thinking off.

        Thinking is off so the prompting strategy is the only variable across cells;
        left on, the model reasons regardless and the chain-of-thought strategy stops
        being a distinct condition. See ModelSpec.send_thinking_disabled for why this
        is a per-model flag rather than a constant.

        Raises:
            RuntimeError: If safety classifiers declined the request.
        """
        extra: dict[str, object] = (
            {"thinking": {"type": "disabled"}} if self._send_thinking_disabled else {}
        )
        response = await self._client.messages.create(
            model=self.model_id,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": t.role.value, "content": t.content} for t in turns],
            **extra,  # pyright: ignore[reportArgumentType]
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"{self.model_id} declined the request")
        return "".join(b.text for b in response.content if b.type == "text")


class OpenAICompatibleProvider:
    """Any model behind an OpenAI-compatible chat-completions endpoint.

    Moonshot (Kimi), OpenAI, and Groq all expose this interface, so swapping model
    families is a config edit in MODELS_UNDER_TEST rather than a new class.
    """

    def __init__(self, spec: ModelSpec, client: openai.AsyncOpenAI) -> None:
        self.model_id = spec.model_id
        self.family = spec.family
        self._extra_body = spec.extra_body
        self._client = client

    async def complete(self, system: str, turns: Sequence[Turn]) -> str:
        """Send one chat-completions request."""
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages += [{"role": t.role.value, "content": t.content} for t in turns]
        response = await self._client.chat.completions.create(
            model=self.model_id,
            max_completion_tokens=MAX_TOKENS,
            messages=messages,  # pyright: ignore[reportArgumentType] - SDK wants TypedDicts
            extra_body=self._extra_body,
        )
        return response.choices[0].message.content or ""


def build_client(spec: ModelSpec) -> openai.AsyncOpenAI:
    """Construct an OpenAI-compatible client for one model spec.

    Args:
        spec: The model to reach, supplying base URL and API-key env var name.

    Returns:
        A client pointed at that provider.

    Raises:
        RuntimeError: If the spec's API key environment variable is unset.
    """
    key = os.environ.get(spec.api_key_env)
    if not key:
        raise RuntimeError(f"{spec.api_key_env} is not set (needed for {spec.model_id})")
    return openai.AsyncOpenAI(api_key=key, base_url=spec.base_url)
