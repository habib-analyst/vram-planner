"""Feasibility planner: turns the estimator into actionable training configs.

Given a model, a target GPU, and a desired *effective* batch size, the planner:

  1. picks the smallest-memory quantization that fits at batch=1
     (tried in order: 4bit -> 8bit -> bf16),
  2. binary-searches the largest per-device micro-batch that fits,
  3. derives gradient-accumulation steps to reach the target effective batch,
  4. decides gradient checkpointing on/off (off only if it still fits — faster),
  5. suggests the 8-bit optimizer when VRAM is tight,
  6. emits a copy-pasteable Unsloth/TRL-style config.
"""

import math

from .estimator import ACTIVATION_K_CHECKPOINT, estimate, format_table
from .gpus_db import get_gpu

QUANT_PREFERENCE = ["4bit", "8bit", "bf16"]
MAX_BATCH_SEARCH = 128


def _fits(model_slug, quant, rank, targets, batch, seq_len, grad_checkpoint, optimizer,
          gpu_slug, utilization: float = 1.0) -> bool:
    bd = estimate(
        model_slug, quant=quant, lora_rank=rank, lora_targets=targets,
        batch=batch, seq_len=seq_len, grad_checkpoint=grad_checkpoint, optimizer=optimizer,
    )
    return bd.total_gb <= get_gpu(gpu_slug)["vram_gb"] * utilization


def recommend_quantization(model_slug, rank=16, targets="attention", seq_len=2048,
                           optimizer="adamw_8bit", gpu_slug="kaggle-t4") -> str | None:
    """Smallest-memory quant that fits at batch=1, or None if nothing fits."""
    for quant in QUANT_PREFERENCE:
        if _fits(model_slug, quant, rank, targets, 1, seq_len, True, optimizer, gpu_slug):
            return quant
    return None


def max_batch_size(model_slug, quant, rank=16, targets="attention", seq_len=2048,
                   grad_checkpoint=True, optimizer="adamw_8bit", gpu_slug="kaggle-t4",
                   hi: int = MAX_BATCH_SEARCH, utilization: float = 0.90) -> int:
    """Largest micro-batch that fits on the GPU (binary search). 0 if batch=1 doesn't fit.

    *utilization* caps recommended usage at a fraction of total VRAM (default
    0.90) so the plan keeps a safety headroom instead of riding the OOM edge.
    """
    def ok(batch):
        return _fits(model_slug, quant, rank, targets, batch, seq_len,
                     grad_checkpoint, optimizer, gpu_slug, utilization)
    if not ok(1):
        return 0
    lo, best, high = 1, 1, hi
    while lo <= high:
        mid = (lo + high) // 2
        if ok(mid):
            best, lo = mid, mid + 1
        else:
            high = mid - 1
    return best


def plan(model_slug: str, target_effective_batch: int = 32, lora_rank: int = 16,
         lora_targets="attention", seq_len: int = 2048, gpu_slug: str = "kaggle-t4") -> dict:
    """Build a full feasibility plan. Raises ValueError if nothing fits at batch=1."""
    gpu = get_gpu(gpu_slug)
    quant = recommend_quantization(model_slug, rank=lora_rank, targets=lora_targets,
                                   seq_len=seq_len, gpu_slug=gpu_slug)
    if quant is None:
        raise ValueError(
            f"{model_slug} does not fit on {gpu['name']} at batch=1 even in 4-bit. "
            "Try a smaller model."
        )

    optimizer = "adamw_8bit" if quant in ("4bit", "8bit") else "adamw"
    micro = max_batch_size(model_slug, quant, rank=lora_rank, targets=lora_targets,
                           seq_len=seq_len, grad_checkpoint=True,
                           optimizer=optimizer, gpu_slug=gpu_slug)
    grad_accum = max(1, math.ceil(target_effective_batch / micro))
    effective = micro * grad_accum

    # Checkpointing: turn it off only if the micro-batch still fits without it.
    ckpt_off_fits = _fits(model_slug, quant, lora_rank, lora_targets, micro, seq_len,
                          False, optimizer, gpu_slug)
    grad_checkpoint = not ckpt_off_fits

    bd = estimate(model_slug, quant=quant, lora_rank=lora_rank, lora_targets=lora_targets,
                  batch=micro, seq_len=seq_len, grad_checkpoint=grad_checkpoint,
                  optimizer=optimizer)
    headroom_gb = gpu["vram_gb"] - bd.total_gb
    tight = headroom_gb < 1.0

    notes = []
    notes.append(f"Smallest feasible quantization: {quant} (fits at batch=1 on {gpu['name']}).")
    notes.append(f"Max micro-batch: {micro} -> grad-accum {grad_accum} = effective batch {effective} "
                 f"(target was {target_effective_batch}).")
    notes.append("Gradient checkpointing " + ("ON (needed to fit)." if grad_checkpoint
                 else "OFF (fits without it — faster training)."))
    if tight:
        notes.append(f"Only {headroom_gb:.2f} GiB headroom — 8-bit optimizer recommended "
                     "(already applied)" if optimizer == "adamw_8bit" else
                     f"Only {headroom_gb:.2f} GiB headroom — consider --optimizer adamw_8bit.")
    else:
        notes.append(f"{headroom_gb:.2f} GiB headroom — comfortable.")
    notes.append("Estimates, not guarantees: verify with a short dry-run on the real GPU.")

    config = {
        "model": model_slug,
        "quantization": quant,
        "load_in_4bit": quant == "4bit",
        "load_in_8bit": quant == "8bit",
        "lora_r": lora_rank,
        "lora_alpha": lora_rank,          # common default: alpha == rank
        "lora_target_modules": bd.lora_targets,
        "per_device_train_batch_size": micro,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": effective,
        "max_seq_length": seq_len,
        "gradient_checkpointing": grad_checkpoint,
        "optimizer": optimizer,
        "learning_rate": 2e-4,
        "fp16": quant == "fp16",
        "bf16": quant in ("bf16", "4bit", "8bit"),
        "estimated_vram_gb": round(bd.total_gb, 2),
        "gpu": gpu_slug,
    }
    return {"config": config, "breakdown": bd, "notes": notes,
            "activation_note": f"activation heuristic K={ACTIVATION_K_CHECKPOINT} (checkpointed)"}


def format_plan(result: dict) -> str:
    """Render the plan as a human-readable report + copy-pasteable config."""
    cfg = result["config"]
    bd = result["breakdown"]
    from .gpus_db import get_gpu
    gpu = get_gpu(cfg["gpu"])

    lines = ["=" * 60, "VRAM PLAN", "=" * 60]
    lines.append(format_table(bd, cfg["gpu"]))
    lines.append("")
    lines.append("Recommendations:")
    for n in result["notes"]:
        lines.append(f"  - {n}")
    lines.append("")
    lines.append("Copy-pasteable config (Unsloth / TRL SFTTrainer style):")
    lines.append("-" * 60)
    lines.append("model, tokenizer = FastLanguageModel.from_pretrained(")
    lines.append(f'    model_name = "{cfg["model"]}",')
    lines.append(f"    load_in_4bit = {cfg['load_in_4bit']},")
    lines.append(f"    load_in_8bit = {cfg['load_in_8bit']},")
    lines.append(")")
    lines.append("model = FastLanguageModel.get_peft_model(")
    lines.append("    model,")
    lines.append(f"    r = {cfg['lora_r']},")
    lines.append(f"    lora_alpha = {cfg['lora_alpha']},")
    lines.append(f"    target_modules = {cfg['lora_target_modules']},")
    lines.append(f"    lora_dropout = 0,")
    lines.append(f"    bias = \"none\",")
    lines.append(")")
    lines.append("trainer = SFTTrainer(")
    lines.append("    model = model,")
    lines.append("    train_dataset = dataset,")
    lines.append(f"    max_seq_length = {cfg['max_seq_length']},")
    lines.append("    args = TrainingArguments(")
    lines.append(f"        per_device_train_batch_size = {cfg['per_device_train_batch_size']},")
    lines.append(f"        gradient_accumulation_steps = {cfg['gradient_accumulation_steps']},  # effective batch {cfg['effective_batch_size']}")
    lines.append(f"        gradient_checkpointing = {cfg['gradient_checkpointing']},")
    lines.append(f"        optim = \"{cfg['optimizer']}\",")
    lines.append(f"        learning_rate = {cfg['learning_rate']},")
    lines.append(f"        fp16 = {cfg['fp16']}, bf16 = {cfg['bf16']},")
    lines.append("        logging_steps = 10,")
    lines.append("    ),")
    lines.append(")")
    lines.append(f"# est. {cfg['estimated_vram_gb']} GiB on {gpu['name']} ({gpu['vram_gb']:.0f} GiB)")
    return "\n".join(lines)
