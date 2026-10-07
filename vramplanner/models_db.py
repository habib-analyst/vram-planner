"""Bundled model database (offline, no network required).

All figures are approximate, compiled from public model cards / config files
(hidden size, layer count) and widely reported parameter counts. They are
accurate enough for VRAM *estimation* (see README calibration notes) but are
NOT authoritative specs — always check the official model card.

Conventions:
  params : total parameter count (int)
  hidden : transformer hidden size (d_model)
  layers : number of transformer blocks
  arch   : "dense" or "moe"
"""

MODELS = {
    # ---- Meta Llama ----
    "llama-3.2-1b": {"name": "Llama 3.2 1B", "params": 1_235_814_400, "hidden": 2048, "layers": 16, "arch": "dense"},
    "llama-3.2-3b": {"name": "Llama 3.2 3B", "params": 3_212_749_568, "hidden": 3072, "layers": 28, "arch": "dense"},
    "llama-3.1-8b": {"name": "Llama 3.1 8B", "params": 8_030_261_248, "hidden": 4096, "layers": 32, "arch": "dense"},
    "llama-3.1-70b": {"name": "Llama 3.1 70B", "params": 70_600_000_000, "hidden": 8192, "layers": 80, "arch": "dense"},
    "llama-3.3-70b": {"name": "Llama 3.3 70B", "params": 70_600_000_000, "hidden": 8192, "layers": 80, "arch": "dense"},
    # ---- Mistral ----
    "mistral-7b-v0.3": {"name": "Mistral 7B v0.3", "params": 7_250_000_000, "hidden": 4096, "layers": 32, "arch": "dense"},
    "mixtral-8x7b": {"name": "Mixtral 8x7B (MoE)", "params": 46_700_000_000, "hidden": 4096, "layers": 32, "arch": "moe"},
    # ---- Qwen2.5 ----
    "qwen2.5-0.5b": {"name": "Qwen2.5 0.5B", "params": 494_000_000, "hidden": 896, "layers": 24, "arch": "dense"},
    "qwen2.5-1.5b": {"name": "Qwen2.5 1.5B", "params": 1_540_000_000, "hidden": 1536, "layers": 28, "arch": "dense"},
    "qwen2.5-3b": {"name": "Qwen2.5 3B", "params": 3_090_000_000, "hidden": 2048, "layers": 36, "arch": "dense"},
    "qwen2.5-7b": {"name": "Qwen2.5 7B", "params": 7_610_000_000, "hidden": 3584, "layers": 28, "arch": "dense"},
    "qwen2.5-14b": {"name": "Qwen2.5 14B", "params": 14_770_000_000, "hidden": 5120, "layers": 48, "arch": "dense"},
    "qwen2.5-32b": {"name": "Qwen2.5 32B", "params": 32_500_000_000, "hidden": 5120, "layers": 64, "arch": "dense"},
    "qwen2.5-72b": {"name": "Qwen2.5 72B", "params": 72_700_000_000, "hidden": 8192, "layers": 80, "arch": "dense"},
    # ---- Qwen3 ----
    "qwen3-8b": {"name": "Qwen3 8B", "params": 8_190_000_000, "hidden": 4096, "layers": 36, "arch": "dense"},
    "qwen3-32b": {"name": "Qwen3 32B", "params": 32_500_000_000, "hidden": 5120, "layers": 64, "arch": "dense"},
    # ---- Google Gemma ----
    "gemma-2-2b": {"name": "Gemma 2 2B", "params": 2_610_000_000, "hidden": 2304, "layers": 26, "arch": "dense"},
    "gemma-2-9b": {"name": "Gemma 2 9B", "params": 9_240_000_000, "hidden": 3584, "layers": 42, "arch": "dense"},
    "gemma-2-27b": {"name": "Gemma 2 27B", "params": 27_200_000_000, "hidden": 4608, "layers": 46, "arch": "dense"},
    "gemma-3-4b": {"name": "Gemma 3 4B", "params": 4_300_000_000, "hidden": 2560, "layers": 34, "arch": "dense"},
    "gemma-3-12b": {"name": "Gemma 3 12B", "params": 12_200_000_000, "hidden": 3840, "layers": 48, "arch": "dense"},
    "gemma-3-27b": {"name": "Gemma 3 27B", "params": 27_400_000_000, "hidden": 5376, "layers": 62, "arch": "dense"},
    # ---- Microsoft Phi ----
    "phi-3-mini": {"name": "Phi-3 Mini 3.8B", "params": 3_820_000_000, "hidden": 3072, "layers": 32, "arch": "dense"},
    "phi-3-medium": {"name": "Phi-3 Medium 14B", "params": 14_000_000_000, "hidden": 5120, "layers": 40, "arch": "dense"},
    "phi-4": {"name": "Phi-4 14B", "params": 14_660_000_000, "hidden": 5120, "layers": 40, "arch": "dense"},
    # ---- Distills / other ----
    "deepseek-r1-distill-qwen-7b": {"name": "DeepSeek-R1-Distill-Qwen 7B", "params": 7_600_000_000, "hidden": 3584, "layers": 28, "arch": "dense"},
    "falcon3-7b": {"name": "Falcon3 7B", "params": 7_500_000_000, "hidden": 3072, "layers": 28, "arch": "dense"},
}

# Common LoRA target-module presets (module names as in HF transformers).
LORA_TARGET_PRESETS = {
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
    "all-linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def get_model(slug: str) -> dict:
    """Return the model record for *slug* (KeyError with a helpful message if unknown)."""
    try:
        return MODELS[slug]
    except KeyError:
        known = ", ".join(sorted(MODELS))
        raise KeyError(f"unknown model '{slug}'. Known models: {known}")


def list_models() -> list:
    return sorted(MODELS)
