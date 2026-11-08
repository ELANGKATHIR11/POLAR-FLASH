"""
POLAR-FLASH: .pflash Container & Streaming Test Suite
Validates Part VI Section 21 (.pflash binary format) and Part III Section 11 (Double-buffering).
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import torch
import numpy as np

from polarflash_core.formats.pflash_format import PFlashWriter, PFlashReader
from polarflash_rt.streaming.flash_streamer import DoubleBufferPipeline


def test_pflash_serialization_and_mmap(tmp_path):
    test_filepath = str(tmp_path / "scratch_test_model.pflash")
    writer = PFlashWriter(page_alignment=4096)
    writer.set_metadata({
        "model_architecture": "qwen2.5-coder-3b-polarflash",
        "total_params": 3_115_872_256,
        "quant_type": "polar_e8_hybrid"
    })

    # Create dummy tensor payloads
    t1_bytes = b"\x01\x02\x03\x04" * 1024  # 4096 bytes (aligned)
    t2_bytes = b"\x0A\x0B\x0C\x0D" * 500   # 2000 bytes (unaligned)

    writer.add_tensor("layer_0.attn.q_proj.weight", t1_bytes, shape=[1024, 4], dtype="int4")
    writer.add_tensor("layer_0.attn.v_proj.weight", t2_bytes, shape=[500, 4], dtype="int4")

    save_info = writer.save(test_filepath)
    assert os.path.exists(test_filepath)
    assert save_info["num_tensors"] == 2
    print(f"[PASS] Serialized .pflash file: {save_info['total_bytes']} bytes, alignment {save_info['page_alignment']}")

    # Read back with zero-copy mmap
    with PFlashReader(test_filepath) as reader:
        assert reader.metadata["model_architecture"] == "qwen2.5-coder-3b-polarflash"
        assert "layer_0.attn.q_proj.weight" in reader.tensor_catalog
        assert "layer_0.attn.v_proj.weight" in reader.tensor_catalog

        read_t1 = reader.get_tensor_bytes("layer_0.attn.q_proj.weight", verify_checksum=True)
        assert read_t1 == t1_bytes
        read_t2 = reader.get_tensor_bytes("layer_0.attn.v_proj.weight", verify_checksum=True)
        assert read_t2 == t2_bytes

    # Cleanup test artifact
    if os.path.exists(test_filepath):
        os.remove(test_filepath)
    print("[PASS] Zero-copy Memory-Mapped .pflash read & CRC32 verification passed!")


def test_double_buffering_pipeline():
    pipeline = DoubleBufferPipeline(io_bandwidth_gb_s=5.0, compute_tflops=110.0)
    # Simulate 36 transformer layers
    layers = [
        {"layer_id": f"layer_{i}", "tensor_bytes": 45 * (1024 ** 2), "flops": 1.2e11}
        for i in range(36)
    ]
    res = pipeline.simulate_pipeline(layers)
    assert res["double_buffer_speedup"] > 1.0
    print(f"[PASS] Double-Buffering Pipeline speedup = {res['double_buffer_speedup']:.2f}x (Serial: {res['serial_time_s'] * 1000:.1f}ms, Pipelined: {res['pipelined_time_s'] * 1000:.1f}ms)")


if __name__ == "__main__":
    test_pflash_serialization_and_mmap()
    test_double_buffering_pipeline()

