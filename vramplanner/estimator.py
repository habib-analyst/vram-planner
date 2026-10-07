"""VRAM estimation model for LoRA / QLoRA fine-tuning.

Component model (all sizes in bytes, reported in GiB = bytes / 2**30):

  weights      = params * bytes_per_param(quant)
  adapters     = trainable_lora_params * 2            (bf16 adapters)
  gradients    = trainable_lora_params * 2            (bf16 grads)
  optimizer    = trainable_lora_params * opt_bytes    (trainable params only)
  activations  = batch * seq_len * hidden * layers * K
                 K = 34 without gradient checkpointing,
                 K =  4 with gradient checkpointing (heuristic)
  cuda_reserve = 1.2 GiB (context, kernels, allocator bookkeeping)
  total        = (sum of the above) * 1.10            (10% fragmentation margin)

bytes_per_param table:
  fp32 4.0 | fp16/bf16 2.0 | int8 1.05 (bitsandbytes overhead)
  | int4 0.55 (NF4 + double quantization, effective)

Optimizer bytes per trainable param:
  adamw      12  (4 fp32 master + 4 momentum + 4 variance)
  adamw_8bit  6  (4 fp32 master + ~2 8-bit states, bitsandbytes)
  sgd         8  (4 fp32 master + 4 momentum)

LoRA trainable params (approximation, documented in README):
  layers * n_target_modules * rank * 2 * hidden
GQA kv-projection dims and per-module in/out asymmetries are ignored; this
keeps the model offline-simple and errs on the slightly conservative side
for attention-only targets.

Calibration anchors (see README for full discussion, incl. error margin):
  * 7B model in fp16  -> exactly 14.0 GiB weights (by construction).
  * 65B QLoRA r64 (attention targets, b=1, s=2048, ckpt, adamw_8bit)
    -> ~48.9 GiB, matching the QLoRA paper's "65B on a single 48GB GPU".
  * 7B QLoRA r64 b=1 s=2048 -> ~7.6 GiB here vs ~10-12 GiB widely reported
    in the wild; the model is optimistic at small scale (allocator
    fragmentation, kernel workspace, seq packing all vary). Treat totals
    as estimates with a ~+-30% band, not guarantees.
"""

from dataclasses import dataclass, field

from .gpus_db import get_gpu
from .models_db import LORA_TARGET_PRESETS, get_model

GB = 1024 ** 3

BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "8bit": 1.05,   # int8, bitsandbytes-style overhead
    "4bit": 0.55,    # NF4 + double quantization, effective
}

OPTIMIZER_BYTES_PER_PARAM = {
    "adamw": 12.0,
    "adamw_8bit": 6.0,
    "sgd": 8.0,
}

ADAPTER_BYTES_PER_PARAM = 2.0   # LoRA adapters kept in bf16
GRADIENT_BYTES_PER_PARAM = 2.0  # bf16 gradients

ACTIVATION_K_NO_CHECKPOINT = 34.0
ACTIVATION_K_CHECKPOINT = 4.0

CUDA_RESERVE_GB = 1.2
FRAGMENTATION_MARGIN = 0.10


@dataclass
class Breakdown:
    model: str
    quant: str
    lora_rank: int
    lora_targets: list
    trainable_params: int
    batch: int
    seq_len: int
    grad_checkpoint: bool
    optimizer: str
    components_gb: dict = field(default_factory=dict)
    total_gb: float = 0.0

    def fits(self, gpu_slug: str) -> bool:
        gpu = get_gpu(gpu_slug)
        return self.total_gb <= gpu["vram_gb"]

    def fits_all(self) -> dict:
        from .gpus_db import GPUS
        return {slug: self.total_gb <= g["vram_gb"] for slug, g in GPUS.items()}


def bytes_per_param(quant: str) -> float:
    try:
        return BYTES_PER_PARAM[quant]
    except KeyError:
        raise ValueError(f"unknown quant '{quant}'; choose from {sorted(BYTES_PER_PARAM)}")


def resolve_lora_targets(targets) -> list:
    """Accept a preset name ('attention'/'mlp'/'all-linear') or a list of module names."""
    if isinstance(targets, str):
        if "," in targets:  # e.g. "q_proj,v_proj"
            return [t.strip() for t in targets.split(",") if t.strip()]
        try:
            return list(LORA_TARGET_PRESETS[targets])
        except KeyError:
            raise ValueError(
                f"unknown LoRA target preset '{targets}'; "
                f"choose from {sorted(LORA_TARGET_PRESETS)} or pass a list"
            )
    return list(targets)


def lora_trainable_params(model_slug: str, rank: int, targets="attention") -> int:
    """Approximate LoRA trainable parameter count.

    layers * n_modules * rank * (in_dim + out_dim), with in_dim = out_dim = hidden.
    """
    if rank <= 0:
        return 0
    model = get_model(model_slug)
    modules = resolve_lora_targets(targets)
    return model["layers"] * len(modules) * rank * 2 * model["hidden"]


def estimate(
    model_slug: str,
    quant: str = "4bit",
    lora_rank: int = 16,
    lora_targets="attention",
    batch: int = 2,
    seq_len: int = 2048,
    grad_checkpoint: bool = True,
    optimizer: str = "adamw_8bit",
) -> Breakdown:
    """Estimate VRAM for a LoRA/QLoRA training step. Returns an itemized Breakdown."""
    model = get_model(model_slug)
    bpp = bytes_per_param(quant)
    targets = resolve_lora_targets(lora_targets)
    trainable = lora_trainable_params(model_slug, lora_rank, targets)

    weights_gb = model["params"] * bpp / GB
    adapters_gb = trainable * ADAPTER_BYTES_PER_PARAM / GB
    grads_gb = trainable * GRADIENT_BYTES_PER_PARAM / GB
    try:
        opt_bpp = OPTIMIZER_BYTES_PER_PARAM[optimizer]
    except KeyError:
        raise ValueError(f"unknown optimizer '{optimizer}'; choose from {sorted(OPTIMIZER_BYTES_PER_PARAM)}")
    optimizer_gb = trainable * opt_bpp / GB

    k = ACTIVATION_K_CHECKPOINT if grad_checkpoint else ACTIVATION_K_NO_CHECKPOINT
    activations_gb = batch * seq_len * model["hidden"] * model["layers"] * k / GB

    components = {
        "weights": weights_gb,
        "lora_adapters": adapters_gb,
        "gradients": grads_gb,
        "optimizer": optimizer_gb,
        "activations": activations_gb,
        "cuda_reserve": CUDA_RESERVE_GB,
    }
    subtotal = sum(components.values())
    total = subtotal * (1.0 + FRAGMENTATION_MARGIN)
    components["fragmentation_margin_10pct"] = total - subtotal

    return Breakdown(
        model=model_slug,
        quant=quant,
        lora_rank=lora_rank,
        lora_targets=targets,
        trainable_params=trainable,
        batch=batch,
        seq_len=seq_len,
        grad_checkpoint=grad_checkpoint,
        optimizer=optimizer,
        components_gb=components,
        total_gb=total,
    )


def format_table(bd: Breakdown, gpu_slug: str | None = None) -> str:
    """Render an itemized VRAM breakdown table, optionally with a fits/doesn't-fit verdict."""
    lines = []
    model = get_model(bd.model)
    lines.append(f"Model: {model['name']} ({model['params'] / 1e9:.2f}B params, {bd.quant})")
    lines.append(
        f"LoRA: rank={bd.lora_rank}, targets={'+'.join(bd.lora_targets)} "
        f"({bd.trainable_params / 1e6:.1f}M trainable), "
        f"batch={bd.batch}, seq_len={bd.seq_len}, "
        f"grad_checkpoint={'on' if bd.grad_checkpoint else 'off'}, optim={bd.optimizer}"
    )
    lines.append("-" * 52)
    lines.append(f"{'component':<28}{'GiB':>10}")
    lines.append("-" * 52)
    for name, gb in bd.components_gb.items():
        lines.append(f"{name:<28}{gb:>10.2f}")
    lines.append("-" * 52)
    lines.append(f"{'TOTAL (est.)':<28}{bd.total_gb:>10.2f}")
    if gpu_slug:
        gpu = get_gpu(gpu_slug)
        verdict = "FITS ✓" if bd.fits(gpu_slug) else "DOES NOT FIT ✗"
        lines.append(f"{verdict} on {gpu['name']} ({gpu['vram_gb']:.0f} GiB)")
    return "\n".join(lines)
