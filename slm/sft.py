from __future__ import annotations

import importlib
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from slm.training import ChatRow, LossPoint, TrainConfig, TrainResult

# The training stack ships partial type information, which under pyright strict produces
# dozens of "partially unknown" errors that hide real ones without catching anything. These
# are imported through an explicitly-Any facade instead: the checker stops guessing, and the
# alternative - one narrow ignore per attribute access - would be twenty suppressions saying
# the same thing. Imported lazily so the ablation and data-generation paths never pay for a
# torch import.
_STACK = ("torch", "datasets", "peft", "transformers", "trl")


def _import_training_stack() -> dict[str, Any]:
    """Import the heavy training dependencies.

    Returns:
        The imported modules, keyed by name.

    Raises:
        ImportError: If any of them is missing.
    """
    try:
        return {name: importlib.import_module(name) for name in _STACK}
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "training needs the train extra: uv pip install -e '.[train]'"
        ) from exc

# Every linear projection in the block. Adapting attention only is the common shortcut;
# the MLP is where a small model stores output-format habits, which is the entire target
# behavior here.
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def render_rows(rows: Sequence[ChatRow], tokenizer: Any) -> list[dict[str, str]]:
    """Split chat rows into the prompt/completion pair TRL masks loss against.

    The prompt is rendered with `add_generation_prompt=True` and thinking disabled -
    byte-identical to what `eval.py` and the demo Space send at inference. Qwen3's template
    closes an empty `<think></think>` pair onto a generation prompt, so splitting here
    rather than rendering the whole conversation is what keeps that scaffold inside the
    masked prefix instead of inside the loss.

    Args:
        rows: `{"messages": [system, user, assistant]}` records.
        tokenizer: The base model's tokenizer, supplying the chat template.

    Returns:
        One `{"prompt": ..., "completion": ...}` record per row.
    """
    rendered: list[dict[str, str]] = []
    for row in rows:
        messages = row["messages"]
        prompt: str = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rendered.append({"prompt": prompt, "completion": str(messages[-1]["content"])})
    return rendered


def run_sft(
    config: TrainConfig,
    rows: Sequence[ChatRow],
    output_dir: Path,
    device: str | None = None,
    load_in_4bit: bool = False,
) -> TrainResult:
    """Fine-tune one curve point and write the merged model to disk.

    Backend-agnostic on purpose: the same call trains on a CUDA box with a 4-bit base
    (true QLoRA) or on Apple Silicon with a bf16 base (LoRA, because bitsandbytes has no
    Metal backend). Only `load_in_4bit` differs, so a run done locally can be reproduced
    on a GPU host without touching the recipe.

    Args:
        config: Hyperparameters, held fixed across every curve point.
        rows: Chat-format training rows.
        output_dir: Where to write the merged model and the adapter.
        device: Torch device, or None to autodetect.
        load_in_4bit: Quantize the frozen base to 4-bit NF4. Requires bitsandbytes on CUDA.

    Returns:
        The finished run's metrics and logs, with `revision` still unset - pushing is the
        caller's job.

    Raises:
        ImportError: If the training extras are not installed.
    """
    stack = _import_training_stack()
    torch, datasets, peft, transformers, trl = (stack[name] for name in _STACK)

    from slm.local import resolve_device

    resolved = resolve_device(device)
    # bf16 halves the memory traffic that dominates on unified memory, and LoRA is robust
    # to it. CPU keeps fp32 because bf16 matmuls there fall back to slow kernels.
    dtype = torch.float32 if resolved == "cpu" else torch.bfloat16

    tokenizer: Any = transformers.AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {"dtype": dtype}
    if load_in_4bit:
        load_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model: Any = transformers.AutoModelForCausalLM.from_pretrained(
        config.base_model, **load_kwargs
    )
    if not load_in_4bit:
        model = model.to(resolved)

    dataset: Any = datasets.Dataset.from_list(render_rows(rows, tokenizer))
    sample_text = f"{dataset[0]['prompt']}{dataset[0]['completion']}"
    print("--- first training example, verbatim ---")
    print(sample_text)
    print("--- end (everything before the reply is masked out of the loss) ---")

    args: Any = trl.SFTConfig(
        output_dir=str(output_dir / "trainer"),
        # Prompt/completion columns plus this flag is what confines the loss to the tutor's
        # reply. Training on the prompt too would spend a 0.6B model's capacity learning to
        # reproduce student code - the one thing the spec forbids it from emitting.
        completion_only_loss=True,
        max_length=config.max_seq_length,
        packing=False,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        # adamw_8bit is bitsandbytes, i.e. CUDA only. The optimizer states here cover the
        # adapter alone, so the plain implementation costs nothing worth optimizing.
        optim="adamw_torch",
        seed=config.seed,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        dataloader_pin_memory=False,
    )
    trainer: Any = trl.SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft.LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ),
    )

    started = time.perf_counter()
    stats: Any = trainer.train()
    runtime = time.perf_counter() - started

    history: list[dict[str, Any]] = list(trainer.state.log_history)
    loss_curve = [
        LossPoint(
            step=int(entry["step"]), epoch=float(entry["epoch"]), loss=float(entry["loss"])
        )
        for entry in history
        if "loss" in entry
    ]

    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged"
    trained: Any = trainer.model
    trained.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    # Merge so the published checkpoint is a plain causal LM: a grader loads it with
    # AutoModelForCausalLM and needs neither peft nor the base repo.
    merged: Any = trained.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    import importlib.metadata as md

    versions: dict[str, str] = {}
    for name in ("torch", "transformers", "trl", "peft", "datasets"):
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:  # pragma: no cover - all are hard deps
            continue
    versions["device"] = resolved
    versions["quantized_base"] = "4bit-nf4" if load_in_4bit else "bf16"

    return TrainResult(
        repo_id=config.repo_id,
        dataset_size=config.dataset_size,
        n_examples=len(rows),
        steps=int(trainer.state.global_step),
        train_loss=float(stats.training_loss),
        runtime_seconds=runtime,
        loss_curve=loss_curve,
        trainer_state={
            "log_history": history,
            "global_step": trainer.state.global_step,
        },
        versions=versions,
        sample_text=sample_text,
        merged_dir=str(merged_dir),
        adapter_dir=str(adapter_dir),
    )
