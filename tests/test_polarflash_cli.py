"""
POLAR-FLASH: CLI End-to-End Command Verification
Tests all command line interface entry points:
- inspect
- quantize
- analyze-rate-distortion
- inspect-memory
- benchmark
- benchmark-physics
- run
- finetune
"""

import sys
import subprocess
import pytest
from pathlib import Path


def run_cli_command(*args):
    python_exe = sys.executable
    cmd = [python_exe, "-m", "polarflash_core.cli"] + list(args)
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))
    return res


def test_cli_quantize_and_inspect(tmp_path):
    output_pflash = str(tmp_path / "cli_model.pflash")

    # 1. Quantize synthetic model
    res = run_cli_command("quantize", "synthetic", "--backend", "int4", "--dim", "64", "--out", output_pflash)
    assert res.returncode == 0
    assert "[SUCCESS] Archive saved" in res.stdout

    # 2. Inspect generated archive
    res_insp = run_cli_command("inspect", output_pflash)
    assert res_insp.returncode == 0
    assert "Format Version" in res_insp.stdout
    assert "Tensor Catalog" in res_insp.stdout


def test_cli_rate_distortion_and_memory(tmp_path):
    output_pflash = str(tmp_path / "cli_model.pflash")
    run_cli_command("quantize", "synthetic", "--backend", "polar", "--dim", "64", "--out", output_pflash)

    # analyze-rate-distortion
    res_rd = run_cli_command("analyze-rate-distortion", output_pflash)
    assert res_rd.returncode == 0
    assert "Empirical Weight Entropy" in res_rd.stdout

    # inspect-memory
    res_mem = run_cli_command("inspect-memory", output_pflash)
    assert res_mem.returncode == 0
    assert "PEAK ACTIVE VRAM" in res_mem.stdout


def test_cli_benchmarks_and_run(tmp_path):
    output_pflash = str(tmp_path / "cli_model.pflash")
    run_cli_command("quantize", "synthetic", "--backend", "int4", "--dim", "64", "--out", output_pflash)

    # benchmark
    res_b = run_cli_command("benchmark", output_pflash)
    assert res_b.returncode == 0
    assert "Storage Utilization U" in res_b.stdout

    # benchmark-physics
    res_bp = run_cli_command("benchmark-physics", output_pflash)
    assert res_bp.returncode == 0
    assert "Roofline" in res_bp.stdout

    # run (cpu)
    res_run = run_cli_command("run", output_pflash, "--device", "cpu")
    assert res_run.returncode == 0
    assert "CPU Execution Result" in res_run.stdout

    # finetune
    res_ft = run_cli_command("finetune", output_pflash, "--method", "pv-gradcode")
    assert res_ft.returncode == 0
    assert "PV-GradCode convergence verified" in res_ft.stdout

