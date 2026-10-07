"""Free-tier GPU table (offline, no network required).

Values are community knowledge about free-tier offerings (Kaggle, Colab,
Lightning AI) plus common pay-as-you-go / local references. Cloud providers
change specs and availability without notice — treat the "note" field as a
reminder to verify against the provider's current docs before planning a run.
"""

GPUS = {
    "kaggle-t4": {
        "name": "Kaggle T4 (free)",
        "vram_gb": 16.0,
        "note": "Kaggle free tier: 2x T4, ~30h GPU/week. Verify quota in Kaggle settings.",
    },
    "kaggle-p100": {
        "name": "Kaggle P100 (free)",
        "vram_gb": 16.0,
        "note": "Kaggle free tier: P100 option, ~30h GPU/week. Verify quota in Kaggle settings.",
    },
    "colab-t4": {
        "name": "Colab T4 (free)",
        "vram_gb": 15.0,
        "note": "Colab free tier reports ~15GB usable on T4. Sessions are preemptible; verify in docs.",
    },
    "colab-a100": {
        "name": "Colab A100 40GB (Pro, paid reference)",
        "vram_gb": 40.0,
        "note": "Colab Pro/Pro+ paid tier. Included as a reference target, not free.",
    },
    "lightning-t4": {
        "name": "Lightning AI T4 (free tier)",
        "vram_gb": 16.0,
        "note": "Lightning AI free tier GPU hours. Verify current free-hour policy in docs.",
    },
    "modal-t4": {
        "name": "Modal T4 (pay-as-you-go reference)",
        "vram_gb": 16.0,
        "note": "Modal serverless GPU, billed per second. Reference target, not free.",
    },
    "rtx-4090": {
        "name": "RTX 4090 24GB (local reference)",
        "vram_gb": 24.0,
        "note": "Common local workstation GPU. Reference target.",
    },
    "rtx-3090": {
        "name": "RTX 3090 24GB (local reference)",
        "vram_gb": 24.0,
        "note": "Common local workstation GPU. Reference target.",
    },
}


def get_gpu(slug: str) -> dict:
    try:
        return GPUS[slug]
    except KeyError:
        known = ", ".join(sorted(GPUS))
        raise KeyError(f"unknown GPU '{slug}'. Known GPUs: {known}")


def list_gpus() -> list:
    return sorted(GPUS)
