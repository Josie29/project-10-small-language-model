from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from slm.dataset import TrainingExample, to_chat_messages

# One TRL-ready SFT record. Spelled out rather than left as `dict[str, object]` so callers
# can index into the messages without a cast.
ChatRow = dict[str, list[dict[str, str]]]

# Sized to the pool's measured p95 (377 characters, roughly 130 tokens with the chat
# template on top), not guessed. Doubling it would only pad batches with masked tokens.
MAX_SEQ_LENGTH = 512

# Unsloth/TRL defaults, held fixed across every curve point. The brief is explicit that a
# disappointing model is a data problem, so these are a constant, not a search space.
LORA_RANK = 16
LORA_ALPHA = 16
LEARNING_RATE = 2e-4
EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
SEED = 42

# ChatML boundaries. Qwen3 renders every turn as `<|im_start|>{role}\n...<|im_end|>`, and
# these two markers are what tell Unsloth to mask the prompt so loss lands only on the
# tutor's reply.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


class TrainConfig(BaseModel):
    """One training run. Everything except `dataset_size` is constant across the curve."""

    base_model: str
    dataset_size: int
    repo_id: str
    epochs: int = EPOCHS
    max_seq_length: int = MAX_SEQ_LENGTH
    lora_rank: int = LORA_RANK
    lora_alpha: int = LORA_ALPHA
    learning_rate: float = LEARNING_RATE
    per_device_batch_size: int = PER_DEVICE_BATCH_SIZE
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS
    seed: int = SEED

    @property
    def effective_batch_size(self) -> int:
        """Examples per optimizer step."""
        return self.per_device_batch_size * self.gradient_accumulation_steps

    @property
    def expected_steps(self) -> int:
        """Optimizer steps this run will take.

        Worth reading before launching a curve point: at the small end this number gets
        low enough that the checkpoint is step-starved as much as data-starved, and the
        curve should be reported with that caveat rather than as pure sample efficiency.
        """
        per_epoch = max(1, self.dataset_size // self.effective_batch_size)
        return per_epoch * self.epochs

    @property
    def warmup_steps(self) -> int:
        """Steps of linear warmup, ~5% of the run.

        Expressed as a fraction of this point's own step count rather than a constant, so
        the smallest curve point does not spend a third of its run warming up.
        """
        return max(1, self.expected_steps // 20)


class LossPoint(BaseModel):
    """One logged training step."""

    step: int
    epoch: float
    loss: float


class TrainResult(BaseModel):
    """What one finished training run reports back from the GPU."""

    repo_id: str
    dataset_size: int
    n_examples: int
    steps: int
    train_loss: float
    runtime_seconds: float
    loss_curve: list[LossPoint]
    # Defaults empty because `run.json` deliberately excludes it - the full log history is
    # written beside it as `trainer_state.json`, so reloading a run does not need it.
    trainer_state: dict[str, object] = {}
    # Resolved at build time rather than pinned in advance. Recorded so a rerun can pin
    # against a known-good set instead of whatever resolves months later.
    versions: dict[str, str]
    # The first fully-rendered training string. The single most useful thing to eyeball in
    # the run log: a chat-template mismatch between training and inference is silent
    # everywhere else and ruins the checkpoint.
    sample_text: str
    merged_dir: str
    adapter_dir: str
    # Hub commit sha, filled in after the push. None means the weights exist only on the
    # machine that trained them, which is exactly what a grader cannot reproduce.
    revision: str | None = None


def curve_subset(pool: Sequence[TrainingExample], size: int) -> list[TrainingExample]:
    """Take the first `size` examples by rank.

    Ranks are assigned in `slm/dataset.py` so that every prefix stays balanced across
    concept, code shape, and the clean:adversarial ratio - which is what makes the curve
    points nested and keeps N the only variable between them.

    Args:
        pool: The full ranked pool.
        size: How many examples this curve point trains on.

    Returns:
        The rank-ordered prefix.

    Raises:
        ValueError: If the pool is smaller than the requested size.
    """
    if size > len(pool):
        raise ValueError(f"pool has {len(pool)} examples, cannot take {size}")
    return sorted(pool, key=lambda e: e.rank)[:size]


def sft_rows(examples: Sequence[TrainingExample]) -> list[ChatRow]:
    """Render examples as the chat rows TRL's SFT path consumes.

    Args:
        examples: The curve point's examples.

    Returns:
        One `{"messages": [...]}` row per example, in rank order.
    """
    return [
        {"messages": to_chat_messages(example)}
        for example in sorted(examples, key=lambda e: e.rank)
    ]
