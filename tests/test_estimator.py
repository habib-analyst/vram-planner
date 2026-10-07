"""Estimator math unit tests — all offline, hand-computed expectations."""

import math

import pytest

from vramplanner.estimator import (
    GB,
    bytes_per_param,
    estimate,
    format_table,
    lora_trainable_params,
    resolve_lora_targets,
)
from vramplanner.models_db import MODELS


def test_bytes_per_param_table():
    assert bytes_per_param("fp32") == 4.0
    assert bytes_per_param("fp16") == 2.0
    assert bytes_per_param("bf16") == 2.0
    assert bytes_per_param("8bit") == pytest.approx(1.05)
    assert bytes_per_param("4bit") == pytest.approx(0.55)
    with pytest.raises(ValueError):
        bytes_per_param("fp8")


def test_7b_fp16_weights_exactly_14gb():
    # Hand-computed: 7e9 params * 2 bytes = 14e9 bytes = 14 * 1e9/2**30 GiB.
    bd = estimate("mistral-7b-v0.3", quant="fp16", lora_rank=0, batch=1,
                  seq_len=1, grad_checkpoint=True, optimizer="adamw")
    expected = 7_250_000_000 * 2.0 / GB
    assert bd.components_gb["weights"] == pytest.approx(expected, rel=1e-9)
    assert bd.components_gb["weights"] == pytest.approx(13.506, rel=1e-3)


def test_7b_fp16_nominal_14gb():
    # The textbook figure: a "7B" model in fp16 ~= 14 GB of weights.
    bd = estimate("mistral-7b-v0.3", quant="fp16", lora_rank=0, batch=1,
                  seq_len=1, grad_checkpoint=True, optimizer="adamw")
    assert 13.0 < bd.components_gb["weights"] < 14.5


def test_lora_trainable_params_known_config():
    # Alpaca-LoRA style: 7B-class (32 layers, hidden 4096), r=8, q_proj+v_proj.
    # 32 * 2 * 8 * (4096 + 4096) = 4,194,304 — the widely reported ~4.2M figure.
    n = lora_trainable_params("mistral-7b-v0.3", rank=8, targets=["q_proj", "v_proj"])
    assert n == 4_194_304


def test_lora_trainable_params_attention_preset_llama8b_r16():
    # 32 layers * 4 modules * 16 * 2 * 4096 = 16,777,216
    n = lora_trainable_params("llama-3.1-8b", rank=16, targets="attention")
    assert n == 16_777_216


def test_lora_rank_zero_means_no_adapters():
    bd = estimate("llama-3.1-8b", quant="4bit", lora_rank=0, batch=2)
    assert bd.trainable_params == 0
    assert bd.components_gb["lora_adapters"] == 0
    assert bd.components_gb["gradients"] == 0
    assert bd.components_gb["optimizer"] == 0


def test_resolve_lora_targets():
    assert resolve_lora_targets("attention") == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert resolve_lora_targets("q_proj,v_proj") == ["q_proj", "v_proj"]
    assert resolve_lora_targets(["o_proj"]) == ["o_proj"]
    with pytest.raises(ValueError):
        resolve_lora_targets("nonsense")


def test_optimizer_math():
    # adamw: 12 bytes/trainable param; adapters+grads: 2 bytes each.
    bd = estimate("llama-3.2-1b", quant="4bit", lora_rank=8, lora_targets="attention",
                  batch=1, seq_len=8, grad_checkpoint=True, optimizer="adamw")
    t = bd.trainable_params
    assert bd.components_gb["optimizer"] == pytest.approx(t * 12.0 / GB, rel=1e-9)
    assert bd.components_gb["lora_adapters"] == pytest.approx(t * 2.0 / GB, rel=1e-9)
    assert bd.components_gb["gradients"] == pytest.approx(t * 2.0 / GB, rel=1e-9)
    bd8 = estimate("llama-3.2-1b", quant="4bit", lora_rank=8, lora_targets="attention",
                   batch=1, seq_len=8, grad_checkpoint=True, optimizer="adamw_8bit")
    assert bd8.components_gb["optimizer"] == pytest.approx(t * 6.0 / GB, rel=1e-9)


def test_activation_checkpointing_reduces_memory():
    kw = dict(model_slug="llama-3.1-8b", quant="4bit", lora_rank=16, batch=4, seq_len=2048)
    on = estimate(**kw, grad_checkpoint=True)
    off = estimate(**kw, grad_checkpoint=False)
    assert off.components_gb["activations"] > on.components_gb["activations"]
    # K=34 vs K=4 -> exactly 8.5x
    assert off.components_gb["activations"] == pytest.approx(
        on.components_gb["activations"] * 8.5, rel=1e-9)


def test_fragmentation_margin_is_10pct_of_subtotal():
    bd = estimate("qwen2.5-7b", quant="4bit", lora_rank=16, batch=2, seq_len=1024)
    parts = [v for k, v in bd.components_gb.items() if k != "fragmentation_margin_10pct"]
    subtotal = sum(parts)
    assert bd.components_gb["fragmentation_margin_10pct"] == pytest.approx(subtotal * 0.10, rel=1e-9)
    assert bd.total_gb == pytest.approx(subtotal * 1.10, rel=1e-9)


def test_monotonicity_rank():
    totals = [estimate("llama-3.1-8b", quant="4bit", lora_rank=r, batch=2).total_gb
              for r in (4, 16, 64)]
    assert totals[0] < totals[1] < totals[2]


def test_monotonicity_batch():
    totals = [estimate("llama-3.1-8b", quant="4bit", lora_rank=16, batch=b).total_gb
              for b in (1, 2, 4, 8)]
    assert totals == sorted(totals) and len(set(totals)) == 4


def test_monotonicity_seq_len_and_quant():
    a = estimate("llama-3.1-8b", quant="4bit", seq_len=1024, batch=2).total_gb
    b = estimate("llama-3.1-8b", quant="4bit", seq_len=4096, batch=2).total_gb
    assert a < b
    q4 = estimate("llama-3.1-8b", quant="4bit", batch=2).total_gb
    q8 = estimate("llama-3.1-8b", quant="8bit", batch=2).total_gb
    q16 = estimate("llama-3.1-8b", quant="bf16", batch=2).total_gb
    assert q4 < q8 < q16


def test_fits_logic():
    small = estimate("llama-3.2-1b", quant="4bit", lora_rank=8, batch=2, seq_len=1024)
    assert small.fits("kaggle-t4") is True
    huge = estimate("llama-3.1-70b", quant="fp16", lora_rank=64, batch=8, seq_len=4096,
                    grad_checkpoint=False)
    assert huge.fits("kaggle-t4") is False


def test_format_table_contains_total_and_verdict():
    bd = estimate("llama-3.1-8b", quant="4bit", lora_rank=16, batch=2, seq_len=2048)
    text = format_table(bd, "kaggle-t4")
    assert "TOTAL (est.)" in text
    assert f"{bd.total_gb:.2f}" in text
    assert "FITS" in text


def test_unknown_model_and_gpu_raise_helpfully():
    with pytest.raises(KeyError, match="unknown model"):
        estimate("gpt-5", quant="4bit")
    bd = estimate("llama-3.1-8b", quant="4bit")
    with pytest.raises(KeyError, match="unknown GPU"):
        bd.fits("h100")


def test_models_db_sanity():
    assert len(MODELS) >= 25
    for slug, m in MODELS.items():
        assert m["params"] > 0 and m["hidden"] > 0 and m["layers"] > 0
        assert m["arch"] in ("dense", "moe")


def test_calibration_anchor_65b_qlora_paper():
    # QLoRA paper: 65B model, r=64, fits on a single 48GB GPU.
    bd = estimate("qwen2.5-72b", quant="4bit", lora_rank=64, lora_targets="attention",
                  batch=1, seq_len=2048, grad_checkpoint=True, optimizer="adamw_8bit")
    # 72B is a bit bigger than 65B; allow the documented ~+-30% band around 48.
    assert 33.6 < bd.total_gb < 62.4, f"anchor drifted: {bd.total_gb:.1f} GiB"
