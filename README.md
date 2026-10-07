# vram-planner

[![CI](https://github.com/habib-analyst/vram-planner/actions/workflows/ci.yml/badge.svg)](https://github.com/habib-analyst/vram-planner/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/habib-analyst/vram-planner/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/habib-analyst/vram-planner)

**Know *before* you launch whether your LoRA/QLoRA run fits on your GPU.**

Free-tier GPUs (Kaggle T4 16 GB, Colab T4 ~15 GB) are where most independent
fine-tuning happens — and where `CUDA out of memory` happens most. Nobody tells
you upfront whether (model + quantization + LoRA rank + batch size + sequence
length) fits. `vram-planner` answers that in milliseconds, offline, with an
itemized breakdown — then auto-plans a feasible config (quantization, max
micro-batch, gradient accumulation, checkpointing, optimizer) and emits a
copy-pasteable Unsloth/TRL training config.

## Architecture

```
 models_db.py ─┐
 gpus_db.py  ──┼─► estimator.py ──► planner.py ──► CLI (estimate / plan / demo)
               │       │                │
               │   itemized VRAM    binary-search max batch,
               │   breakdown        grad-accum, ckpt decision,
               │                    copy-pasteable config
               └─ bundled, offline — no network, no API keys
```

## Quickstart (< 5 min)

```bash
pip install -r requirements.txt

# Will it fit? (one config, itemized breakdown)
python -m vramplanner estimate --model llama-3.1-8b --quant 4bit \
    --lora-rank 16 --batch 2 --seq-len 2048 --gpu kaggle-t4

# Auto-plan: give me a feasible config for effective batch 32
python -m vramplanner plan --model qwen2.5-7b --target-batch 32 \
    --lora-rank 16 --gpu kaggle-t4

# Worked example, no arguments needed
python -m vramplanner demo

# What models / GPUs are bundled?
python -m vramplanner list-models
python -m vramplanner list-gpus
```

As an installed CLI (after `pip install .`): `vram-planner estimate ...`

## Real example output

`python -m vramplanner demo` (actual output, v0.1.0):

```
============================================================
vram-planner demo — Llama 3.1 8B QLoRA on a free Kaggle T4
============================================================

Question: can I fine-tune Llama-3.1-8B with QLoRA (r=16)
on a free Kaggle T4 (16 GiB) before launching the run?

Model: Llama 3.1 8B (8.03B params, 4bit)
LoRA: rank=16, targets=q_proj+k_proj+v_proj+o_proj (16.8M trainable), batch=2, seq_len=2048, grad_checkpoint=on, optim=adamw_8bit
----------------------------------------------------
component                          GiB
----------------------------------------------------
weights                           4.11
lora_adapters                     0.03
gradients                         0.03
optimizer                         0.09
activations                       2.00
cuda_reserve                      1.20
fragmentation_margin_10pct        0.75
----------------------------------------------------
TOTAL (est.)                      8.22
FITS ✓ on Kaggle T4 (free) (16 GiB)

Verdict: it fits. Now auto-plan for an effective batch of 16:

============================================================
VRAM PLAN
============================================================
Model: Llama 3.1 8B (8.03B params, 4bit)
LoRA: rank=16, targets=q_proj+k_proj+v_proj+o_proj (16.8M trainable), batch=7, seq_len=2048, grad_checkpoint=on, optim=adamw_8bit
----------------------------------------------------
component                          GiB
----------------------------------------------------
weights                           4.11
lora_adapters                     0.03
gradients                         0.03
optimizer                         0.09
activations                       7.00
cuda_reserve                      1.20
fragmentation_margin_10pct        1.25
----------------------------------------------------
TOTAL (est.)                     13.72
FITS ✓ on Kaggle T4 (free) (16 GiB)

Recommendations:
  - Smallest feasible quantization: 4bit (fits at batch=1 on Kaggle T4 (free)).
  - Max micro-batch: 7 -> grad-accum 3 = effective batch 21 (target was 16).
  - Gradient checkpointing ON (needed to fit).
  - 2.28 GiB headroom — comfortable.
  - Estimates, not guarantees: verify with a short dry-run on the real GPU.

Copy-pasteable config (Unsloth / TRL SFTTrainer style):
------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "llama-3.1-8b",
    load_in_4bit = True,
    load_in_8bit = False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0,
    bias = "none",
)
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 7,
        gradient_accumulation_steps = 3,  # effective batch 21
        gradient_checkpointing = True,
        optim = "adamw_8bit",
        learning_rate = 0.0002,
        fp16 = False, bf16 = True,
        logging_steps = 10,
    ),
)
# est. 13.72 GiB on Kaggle T4 (free) (16 GiB)
```

## The VRAM model

All sizes in GiB (bytes / 2³⁰):

| Component | Formula |
|---|---|
| weights | `params × bytes_per_param(quant)` — fp32 4.0, fp16/bf16 2.0, int8 1.05, int4 (NF4+double-quant) 0.55 |
| lora_adapters | `trainable × 2` (bf16) |
| gradients | `trainable × 2` (bf16) |
| optimizer | `trainable × 12` (AdamW), `× 6` (AdamW 8-bit), `× 8` (SGD) |
| activations | `batch × seq_len × hidden × layers × K`, K=34 (no ckpt) / K=4 (ckpt) |
| cuda_reserve | 1.2 GiB fixed (context, kernels, allocator) |
| fragmentation | +10% on the subtotal |

LoRA trainable params ≈ `layers × n_target_modules × rank × 2 × hidden`
(GQA kv-dim asymmetries ignored — documented approximation).

The planner caps recommendations at **90% of GPU VRAM** so you keep a safety
headroom instead of riding the OOM edge.

## Calibration — how wrong is it?

Honest numbers, not marketing:

- **Anchor 1 (exact):** 7B model in fp16 → 14.0 GiB weights. By construction.
- **Anchor 2 (paper):** the QLoRA paper fine-tuned a 65B model at r=64 on a
  single 48 GB GPU. This estimator gives ~48.9 GiB for that configuration.
  Agreement within ~2%.
- **Anchor 3 (community):** 7B QLoRA r=64, batch 1, seq 2048 → estimator says
  ~7.6 GiB; widely reported real-world usage is ~10–12 GiB. The model is
  **optimistic at small scale** — allocator fragmentation, kernel workspace,
  and sequence packing vary run to run.

**Treat every total as an estimate with a ±25–30% band, not a guarantee.**
The 10% fragmentation margin and 90% planner cap exist precisely because real
GPUs are messier than arithmetic.

## Disclaimer

**Estimates, not guarantees — verify on your GPU.** Model specs in the
bundled DB are approximate (compiled from public model cards); free-tier GPU
specs are community knowledge and providers change quotas without notice.
Always do a short dry-run (a few steps, `nvidia-smi` watching) before
committing a 30-hour Kaggle session to a plan.

## Roadmap

- [ ] Per-module activation model (attention vs MLP split, GQA-aware kv dims)
- [ ] FSDP / multi-GPU sharding estimates
- [ ] `vram-planner watch` — live `nvidia-smi` comparison against the estimate
- [ ] Community-reported actuals DB to tighten calibration per model/GPU pair
- [ ] DoRA / rsLoRA / PiSSA adapter variants

## Citations

- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023) —
  4-bit NF4 + double quantization memory figures; 65B-on-48GB anchor.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021) —
  adapter parameter accounting.
- Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication* (2022) —
  int8 overhead factor.
- Unsloth documentation — QLoRA training defaults (alpha=rank, 8-bit AdamW,
  gradient checkpointing) mirrored in the emitted configs.

## License

MIT — see [LICENSE](LICENSE). © 2026 Habib Ur Rehman.
