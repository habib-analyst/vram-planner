"""CLI tests — all offline (subprocess demo smoke test included)."""

import subprocess
import sys

from vramplanner.cli import build_parser, main


def test_cli_estimate_runs(capsys):
    main(["estimate", "--model", "llama-3.2-1b", "--quant", "4bit",
          "--lora-rank", "8", "--batch", "1", "--seq-len", "512", "--gpu", "kaggle-t4"])
    out = capsys.readouterr().out
    assert "TOTAL (est.)" in out and "FITS" in out


def test_cli_plan_runs(capsys):
    main(["plan", "--model", "qwen2.5-1.5b", "--target-batch", "16",
          "--lora-rank", "8", "--gpu", "kaggle-t4"])
    out = capsys.readouterr().out
    assert "VRAM PLAN" in out and "SFTTrainer" in out


def test_cli_demo_runs(capsys):
    main(["demo"])
    out = capsys.readouterr().out
    assert "Llama 3.1 8B" in out
    assert "Kaggle T4" in out
    assert "FITS" in out


def test_cli_list_commands(capsys):
    main(["list-models"])
    assert "llama-3.1-8b" in capsys.readouterr().out
    main(["list-gpus"])
    assert "kaggle-t4" in capsys.readouterr().out


def test_cli_no_args_runs_demo(capsys):
    main([])
    assert "vram-planner demo" in capsys.readouterr().out


def test_cli_all_gpus_flag(capsys):
    main(["estimate", "--model", "llama-3.2-1b", "--batch", "1",
          "--seq-len", "512", "--all-gpus"])
    out = capsys.readouterr().out
    assert "kaggle-t4" in out or "Kaggle" in out


def test_demo_subprocess_smoke():
    # `python -m vramplanner demo` must work out of the box, no network.
    proc = subprocess.run(
        [sys.executable, "-m", "vramplanner", "demo"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Llama 3.1 8B" in proc.stdout
    assert "FITS" in proc.stdout


def test_parser_rejects_bad_quant():
    parser = build_parser()
    try:
        parser.parse_args(["estimate", "--quant", "fp8"])
    except SystemExit as e:
        assert e.code == 2
    else:  # pragma: no cover
        raise AssertionError("expected argparse to reject --quant fp8")
