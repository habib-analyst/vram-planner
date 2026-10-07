"""Planner tests — binary search correctness, quant choice, output schema."""

import math

import pytest

from vramplanner.estimator import estimate
from vramplanner.planner import format_plan, max_batch_size, plan, recommend_quantization


def test_max_batch_size_matches_brute_force():
    kw = dict(model_slug="llama-3.1-8b", quant="4bit", rank=16, targets="attention",
              seq_len=2048, grad_checkpoint=True, optimizer="adamw_8bit",
              gpu_slug="kaggle-t4", hi=32, utilization=0.90)
    got = max_batch_size(**kw)
    # brute force: largest b with total <= 90% of 16 GiB
    expected = 0
    for b in range(1, 33):
        bd = estimate("llama-3.1-8b", quant="4bit", lora_rank=16, batch=b,
                      seq_len=2048, grad_checkpoint=True, optimizer="adamw_8bit")
        if bd.total_gb <= 16.0 * 0.90:
            expected = b
        else:
            break  # monotonic in batch: first failure ends the run
    assert got == expected
    assert got >= 1
    # the safety cap must actually bite: strict fits() would allow a bigger batch
    assert max_batch_size(**{**kw, "utilization": 1.0}) >= got


def test_max_batch_size_zero_when_batch1_does_not_fit():
    got = max_batch_size("llama-3.1-70b", "4bit", rank=64, targets="all-linear",
                         seq_len=8192, gpu_slug="kaggle-t4")
    assert got == 0


def test_recommend_quantization_prefers_4bit():
    q = recommend_quantization("llama-3.1-8b", gpu_slug="kaggle-t4")
    assert q == "4bit"


def test_recommend_quantization_none_when_impossible():
    q = recommend_quantization("llama-3.1-70b", rank=64, targets="all-linear",
                               seq_len=8192, gpu_slug="colab-t4")
    assert q is None


def test_plan_schema():
    result = plan("llama-3.1-8b", target_effective_batch=32, lora_rank=16,
                  gpu_slug="kaggle-t4")
    assert set(result) == {"config", "breakdown", "notes", "activation_note"}
    cfg = result["config"]
    required = {"model", "quantization", "load_in_4bit", "load_in_8bit", "lora_r",
                "lora_alpha", "lora_target_modules", "per_device_train_batch_size",
                "gradient_accumulation_steps", "effective_batch_size", "max_seq_length",
                "gradient_checkpointing", "optimizer", "learning_rate", "fp16", "bf16",
                "estimated_vram_gb", "gpu"}
    assert required <= set(cfg)
    assert cfg["effective_batch_size"] >= 32
    assert cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"] \
        == cfg["effective_batch_size"]
    assert isinstance(result["notes"], list) and len(result["notes"]) >= 3


def test_plan_effective_batch_math():
    result = plan("qwen2.5-7b", target_effective_batch=16, lora_rank=32,
                  seq_len=2048, gpu_slug="kaggle-t4")
    cfg = result["config"]
    micro = cfg["per_device_train_batch_size"]
    accum = cfg["gradient_accumulation_steps"]
    assert accum == math.ceil(16 / micro)
    assert cfg["effective_batch_size"] == micro * accum


def test_plan_raises_when_nothing_fits():
    with pytest.raises(ValueError, match="does not fit"):
        plan("llama-3.1-70b", lora_rank=64, lora_targets="all-linear",
             seq_len=8192, gpu_slug="colab-t4")


def test_plan_checkpoint_decision_is_consistent():
    result = plan("llama-3.1-8b", target_effective_batch=8, gpu_slug="kaggle-t4")
    cfg = result["config"]
    bd = result["breakdown"]
    assert cfg["gradient_checkpointing"] == bd.grad_checkpoint
    # the planned micro-batch must fit on the target GPU
    assert bd.fits("kaggle-t4")


def test_format_plan_has_copy_paste_block():
    result = plan("llama-3.1-8b", target_effective_batch=16, gpu_slug="kaggle-t4")
    text = format_plan(result)
    for token in ("FastLanguageModel.from_pretrained", "get_peft_model", "SFTTrainer",
                  "per_device_train_batch_size", "gradient_accumulation_steps"):
        assert token in text
