"""
POLAR-FLASH: End-to-End LLM Architecture Benchmark Suite
Audits and compares quantization backends across an open-weight transformer projection:
Backends: FP16, INT8, INT4, INT3, INT2, E8 Lattice, PolarQuant (4-bit, 3-bit, 2-bit, and 1.75bpw target).
Metrics:
- Numerical accuracy: MSE, SNR (dB), Cosine Similarity, Max Absolute Error
- Storage metrics: Serialized byte payload, exact effective bpw
- Physical metrics: Operational Intensity (FLOPs/byte), Attainable TFLOPS, Energy (Joules)
- Hardware execution: Generation throughput (tokens/sec), Latency (ms), VRAM allocation
"""

import os
import sys
import time
import json
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List

from polarflash_quant.unified_quantizer import UnifiedQuantizer
from polarflash_quant.hardware.roofline import (
    RooflineAnalyzer, RTX_5060_LAPTOP, compute_energy_consumption
)
from polarflash_core.formats.pflash_format import PFlashWriter, PFlashReader
from polarflash_rt.cpu.engine import PolarLinearLayer


def benchmark_all_backends(
    in_dim: int = 2048,
    out_dim: int = 2048,
    batch_size: int = 1,
    seq_len: int = 128,
    seed: int = 42
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Synthetic realistic LLM transformer projection weights (e.g. Qwen / Llama q_proj)
    weight = torch.randn(out_dim, in_dim, dtype=torch.float32) * (1.0 / math.sqrt(in_dim))
    # Inject 0.1% heavy activation outliers typical of large language models
    outlier_mask = (torch.rand_like(weight) < 0.001)
    weight[outlier_mask] *= 15.0

    num_elements = weight.numel()
    var_w = float(torch.var(weight).item())

    # Roofline Analyzer
    roofline = RooflineAnalyzer(hardware=RTX_5060_LAPTOP)
    flops_per_token = 2 * in_dim * out_dim
    total_flops = flops_per_token * batch_size * seq_len

    backends_config = [
        ("fp16", {"backend": "fp16", "apply_hadamard": False}),
        ("int8", {"backend": "int8", "apply_hadamard": False}),
        ("int4", {"backend": "int4", "apply_hadamard": True}),
        ("int3", {"backend": "int3", "apply_hadamard": True}),
        ("int2", {"backend": "int2", "apply_hadamard": True}),
        ("e8",   {"backend": "e8", "apply_hadamard": True, "e8_stages": 1}),
        ("polar_4b", {"backend": "polar", "apply_hadamard": True, "polar_r_bits": 4, "polar_theta_bits": 4}),
        ("polar_3b", {"backend": "polar", "apply_hadamard": True, "polar_r_bits": 3, "polar_theta_bits": 3}),
        ("polar_2b", {"backend": "polar", "apply_hadamard": True, "polar_r_bits": 2, "polar_theta_bits": 2}),
    ]

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "device": device,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0
        },
        "layer_config": {
            "in_dim": in_dim,
            "out_dim": out_dim,
            "total_parameters": num_elements,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "total_flops": total_flops
        },
        "backends": {}
    }

    dummy_input = torch.randn(batch_size, seq_len, in_dim)

    for label, cfg in backends_config:
        quantizer = UnifiedQuantizer(**cfg)
        payload, meta, w_hat = quantizer.quantize(weight, name=f"layer_{label}")

        # Numerical Diagnostics
        mse = float(torch.mean((weight - w_hat) ** 2).item())
        max_abs_err = float(torch.max(torch.abs(weight - w_hat)).item())
        snr_db = 10.0 * np.log10(max(var_w, 1e-15) / max(mse, 1e-15))
        cos_sim = float(
            torch.cosine_similarity(weight.flatten().unsqueeze(0), w_hat.flatten().unsqueeze(0)).item()
        )

        effective_bpw = (len(payload) * 8.0) / num_elements
        compression_ratio = (num_elements * 2.0) / len(payload)  # vs FP16

        # Physical Roofline & Operational Intensity
        bytes_transferred = len(payload) + (batch_size * seq_len * (in_dim + out_dim) * 2)  # weights + activations
        roof_res = roofline.analyze_layer(total_flops, bytes_transferred)

        # Physical Energy Consumption Model
        energy_res = compute_energy_consumption(
            num_flops=total_flops,
            bytes_transferred_vram=float(bytes_transferred),
            bytes_transferred_ssd=float(len(payload))
        )

        # Hardware Runtime Benchmark (via PolarLinearLayer)
        layer = PolarLinearLayer(
            in_features=in_dim,
            out_features=out_dim,
            dtype=meta["dtype"],
            quant_meta=meta,
            raw_bytes=payload,
            keep_cached=True
        )

        # Warmup
        _ = layer(dummy_input)

        latencies = []
        for _ in range(15):
            t0 = time.perf_counter()
            _ = layer(dummy_input)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        avg_latency_ms = (sum(latencies) / len(latencies)) * 1000.0
        tokens_per_sec = (batch_size * seq_len) / max(sum(latencies) / len(latencies), 1e-6)

        results["backends"][label] = {
            "effective_bpw": round(effective_bpw, 3),
            "serialized_bytes": len(payload),
            "compression_ratio_vs_fp16": round(compression_ratio, 2),
            "mse_distortion": mse,
            "max_abs_error": max_abs_err,
            "snr_db": round(snr_db, 2),
            "cosine_similarity": round(cos_sim, 5),
            "operational_intensity": round(roof_res["operational_intensity_flops_per_byte"], 2),
            "attainable_tflops": round(roof_res["attainable_tflops"], 2),
            "regime": roof_res["bottleneck"],
            "total_energy_microjoules": round(energy_res["total_energy_joules"] * 1e6, 2),
            "latency_ms": round(avg_latency_ms, 3),
            "tokens_per_sec": round(tokens_per_sec, 1)
        }

    os.makedirs("reports", exist_ok=True)
    report_path = "reports/polarflash_end_to_end_backends.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[SUCCESS] End-to-end benchmark complete. Report saved to {report_path}")
    return results


if __name__ == "__main__":
    benchmark_all_backends()

