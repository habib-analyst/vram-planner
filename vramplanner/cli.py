"""CLI for vram-planner."""

import argparse

from . import __version__
from .estimator import estimate, format_table
from .gpus_db import GPUS, list_gpus
from .models_db import list_models
from .planner import format_plan, plan


def cmd_estimate(args):
    bd = estimate(
        args.model, quant=args.quant, lora_rank=args.lora_rank,
        lora_targets=args.lora_targets, batch=args.batch, seq_len=args.seq_len,
        grad_checkpoint=not args.no_grad_checkpoint, optimizer=args.optimizer,
    )
    print(format_table(bd, args.gpu))
    if args.all_gpus:
        print()
        print(f"{'GPU':<34}{'VRAM':>8}  verdict")
        print("-" * 52)
        for slug in list_gpus():
            g = GPUS[slug]
            mark = "FITS ✓" if bd.fits(slug) else "OOM ✗"
            print(f"{g['name']:<34}{g['vram_gb']:>7.0f}  {mark}")


def cmd_plan(args):
    result = plan(
        args.model, target_effective_batch=args.target_batch, lora_rank=args.lora_rank,
        lora_targets=args.lora_targets, seq_len=args.seq_len, gpu_slug=args.gpu,
    )
    print(format_plan(result))


def cmd_demo(_args):
    print("=" * 60)
    print("vram-planner demo — Llama 3.1 8B QLoRA on a free Kaggle T4")
    print("=" * 60)
    print()
    print("Question: can I fine-tune Llama-3.1-8B with QLoRA (r=16)")
    print("on a free Kaggle T4 (16 GiB) before launching the run?")
    print()
    bd = estimate("llama-3.1-8b", quant="4bit", lora_rank=16, lora_targets="attention",
                  batch=2, seq_len=2048, grad_checkpoint=True, optimizer="adamw_8bit")
    print(format_table(bd, "kaggle-t4"))
    print()
    print("Verdict: it fits. Now auto-plan for an effective batch of 16:")
    print()
    result = plan("llama-3.1-8b", target_effective_batch=16, lora_rank=16,
                  lora_targets="attention", seq_len=2048, gpu_slug="kaggle-t4")
    print(format_plan(result))


def cmd_list_models(_args):
    from .models_db import MODELS
    for slug in list_models():
        m = MODELS[slug]
        print(f"{slug:<32}{m['name']:<28}{m['params'] / 1e9:>7.2f}B")


def cmd_list_gpus(_args):
    for slug in list_gpus():
        g = GPUS[slug]
        print(f"{slug:<16}{g['name']:<36}{g['vram_gb']:>5.0f} GiB  # {g['note']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vram-planner",
        description="LoRA/QLoRA VRAM estimator + free-tier GPU feasibility planner.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command")

    def add_common(sp):
        sp.add_argument("--model", default="llama-3.1-8b",
                        help="model slug (see list-models)")
        sp.add_argument("--quant", default="4bit",
                        choices=["4bit", "8bit", "fp16", "bf16", "fp32"])
        sp.add_argument("--lora-rank", type=int, default=16)
        sp.add_argument("--lora-targets", default="attention",
                        help="preset: attention|mlp|all-linear, or comma-separated modules")
        sp.add_argument("--seq-len", type=int, default=2048)

    e = sub.add_parser("estimate", help="estimate VRAM for one configuration")
    add_common(e)
    e.add_argument("--batch", type=int, default=2)
    e.add_argument("--no-grad-checkpoint", action="store_true")
    e.add_argument("--optimizer", default="adamw_8bit",
                   choices=["adamw", "adamw_8bit", "sgd"])
    e.add_argument("--gpu", default="kaggle-t4", help="GPU slug (see list-gpus)")
    e.add_argument("--all-gpus", action="store_true",
                   help="also show fits/doesn't-fit for every known GPU")
    e.set_defaults(func=cmd_estimate)

    pl = sub.add_parser("plan", help="auto-plan a feasible training config")
    add_common(pl)
    pl.add_argument("--target-batch", type=int, default=32,
                    help="desired effective batch size")
    pl.add_argument("--gpu", default="kaggle-t4")
    pl.set_defaults(func=cmd_plan)

    d = sub.add_parser("demo", help="print a realistic worked example")
    d.set_defaults(func=cmd_demo)

    lm = sub.add_parser("list-models", help="list bundled models")
    lm.set_defaults(func=cmd_list_models)
    lg = sub.add_parser("list-gpus", help="list known GPUs")
    lg.set_defaults(func=cmd_list_gpus)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        cmd_demo(args)  # `python -m vramplanner` runs the demo
    else:
        args.func(args)


if __name__ == "__main__":
    main()
