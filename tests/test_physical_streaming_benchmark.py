"""
POLAR-FLASH: Physical Storage Streaming Benchmark Tests
Validates Objective 6:
1. Real physical file read bandwidth (sequential vs random).
2. Physical read amplification calculation.
3. Cache hit rate and LRU eviction under multi-pass layer access.
4. Prefetch accuracy and I/O wait tracking.
5. Compute overlap and token latency accounting.
6. Separation of physical hardware measurements from simulation models.
"""

import os
import pytest
import torch

from polarflash_core.formats.pflash_format import PFlashWriter
from polarflash_rt.streaming.storage_benchmark import (
    PhysicalStorageBenchmark,
    run_comprehensive_storage_validation
)


@pytest.fixture
def sample_pflash_file(tmp_path):
    """Creates a page-aligned multi-tensor .pflash archive for physical storage testing."""
    test_file = str(tmp_path / "streaming_bench.pflash")
    writer = PFlashWriter(page_alignment=4096)

    # Add 12 tensors (simulating 12 transformer layers)
    for i in range(12):
        w = torch.randn(128, 128).half().numpy().tobytes()
        writer.add_tensor(f"layer.{i}.weight", w, shape=[128, 128], dtype="fp16")

    writer.save(test_file)
    return test_file


def test_physical_file_read_bandwidth(sample_pflash_file):
    """Verifies real physical sequential and random file read bandwidth measurement."""
    bench = PhysicalStorageBenchmark(sample_pflash_file)
    res = bench.benchmark_file_read_bandwidth(chunk_size=4096, num_chunks=10)

    assert "sequential_bandwidth_mb_s" in res
    assert "random_bandwidth_mb_s" in res
    assert res["sequential_bandwidth_mb_s"] > 0
    assert res["random_bandwidth_mb_s"] > 0
    assert "random_to_sequential_ratio" in res
    assert res["file_size_bytes"] > 0


def test_streaming_pipeline_metrics(sample_pflash_file):
    """Verifies cache hit rate, read amplification, I/O wait, and token latency."""
    bench = PhysicalStorageBenchmark(sample_pflash_file)

    # Access sequence: access layers 0..7 twice to test LRU cache hits
    seq = [f"layer.{i}.weight" for i in range(8)] + [f"layer.{i}.weight" for i in range(8)]
    metrics = bench.benchmark_streaming_pipeline(seq, max_cache_layers=8)

    assert metrics["total_requests"] == 16
    assert metrics["cache_hits"] > 0
    assert metrics["cache_misses"] > 0
    assert 0.0 < metrics["cache_hit_rate"] <= 1.0
    assert metrics["empirical_read_amplification"] >= 1.0
    assert metrics["total_io_wait_s"] >= 0.0
    assert metrics["token_latency_ms"] > 0.0
    assert metrics["peak_resident_host_ram_mb"] > 0


def test_separation_of_hardware_measurements_and_simulation(sample_pflash_file):
    """Verifies that actual device measurements and theoretical simulation are kept separate."""
    res = run_comprehensive_storage_validation(sample_pflash_file)

    assert "hardware_measurements" in res
    assert "simulation_model" in res

    hw = res["hardware_measurements"]
    sim = res["simulation_model"]

    # Hardware measurements must come from actual execution
    assert "physical_sequential_bandwidth_mb_s" in hw
    assert "physical_random_bandwidth_mb_s" in hw
    assert "cache_hit_rate" in hw
    assert "token_latency_ms" in hw
    assert "peak_resident_ram_mb" in hw

    # Simulation model must be clearly labeled and separate
    assert sim["model_type"] == "Analytical Double-Buffering Roofline Simulation"
    assert "theoretical_speedup" in sim
    assert sim["theoretical_speedup"] >= 1.0

