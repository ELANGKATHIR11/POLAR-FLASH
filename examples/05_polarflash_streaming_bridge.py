#!/usr/bin/env python3
"""
Example 5: POLAR-FLASH Memory-Mapped Streaming & Inspection Bridge
Demonstrates zero-copy memory-mapped archive streaming, format verification,
effective bpw accounting, and real-time disk-to-RAM throughput benchmarking.
"""

import os
import torch
from polarflash import PolarFlashBridge, PFlashWriter, PFlashReader

def main():
    print("=" * 70)
    print("POLAR-FLASH: .pflash Memory-Mapped Streaming Bridge")
    print("=" * 70)

    # 1. Create a sample .pflash archive with quantized layers
    test_archive = "sample_bridge_model.pflash"
    writer = PFlashWriter()
    writer.set_metadata({
        "source_model": "Sample-SLM",
        "compression_scheme": "polar4",
        "framework": "POLAR-FLASH v1.1.0"
    })

    # Simulate 4 transformer projection matrices (e.g. 1024x1024 each in INT4 packed representation)
    for layer in range(4):
        tensor_name = f"model.layers.{layer}.mlp.down_proj.weight"
        raw_payload = os.urandom(1024 * 512)  # 512 KB packed payload
        writer.add_tensor(
            name=tensor_name,
            data_bytes=raw_payload,
            shape=[1024, 1024],
            dtype="polar4",
            quant_meta={
                "r_bits": 4,
                "theta_bits": 4,
                "effective_nominal_bpw": 4.0,
                "snr_db": 16.5
            }
        )

    writer.save(test_archive)
    print(f"[+] Created sample archive: {test_archive} ({os.path.getsize(test_archive) / (1024**2):.2f} MB)")

    # 2. Inspect archive via PolarFlashBridge
    bridge = PolarFlashBridge(default_pflash_path=test_archive)
    info = bridge.inspect_archive()

    print("\n--- Archive Inspection Report ---")
    print(f"Archive Size: {info['archive_size_mb']} MB")
    print(f"Total Tensors: {info['tensor_count']}")
    print(f"Total Elements: {info['total_weight_elements']:,}")
    print(f"Effective BPW: {info['effective_bpw']} bpw")
    print(f"Compression vs FP16: {info['compression_ratio_vs_fp16']}x")
    print(f"Compression vs FP32: {info['compression_ratio_vs_fp32']}x")

    # 3. Stream a single tensor zero-copy
    t0_name = info["tensors"][0]["name"]
    t0_bytes = bridge.stream_tensor(t0_name)
    print(f"\n[+] Streamed tensor '{t0_name}': {len(t0_bytes)} bytes retrieved via mmap")

    # 4. Benchmark Streaming Throughput
    bench = bridge.benchmark_streaming_throughput(iterations=10)
    print("\n--- Streaming Throughput Benchmark ---")
    print(f"Iterations: {bench['iterations']}")
    print(f"Total Data Read: {bench['total_mb_read']} MB")
    print(f"Elapsed Time: {bench['elapsed_seconds']}s")
    print(f"Throughput: {bench['throughput_mb_per_sec']} MB/s (Direct zero-copy mmap)")

    # Cleanup temporary file
    if os.path.exists(test_archive):
        os.remove(test_archive)
    print("\n[+] Demo completed successfully!")

if __name__ == "__main__":
    main()
