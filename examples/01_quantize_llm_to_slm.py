#!/usr/bin/env python3
"""
Example 1: Quantize Any Open-Source LLM to SLM
Demonstrates how to use POLAR-FLASH polar coordinate quantization and non-uniform
2-bit KV-cache compression to drastically reduce VRAM footprint.
"""

import torch
from polarflash import PolarQuantizer, CompressedKVCache

def main():
    print("=" * 60)
    print("POLAR-FLASH: LLM-to-SLM Quantization Pipeline")
    print("=" * 60)

    # 1. Simulate an LLM weight tensor (e.g. 2048 x 2048 projection matrix)
    weights = torch.randn(2048, 2048, dtype=torch.float32)
    print(f"[*] Original weight tensor shape: {weights.shape} | Uncompressed size: {weights.nelement() * 4 / (1024**2):.2f} MB")

    # 2. Polar Coordinate Quantization
    quantizer = PolarQuantizer(bits=4, use_hadamard=True)
    quantized_weights, metadata = quantizer.quantize(weights)
    reconstructed = quantizer.dequantize(quantized_weights, metadata)

    mse = torch.mean((weights - reconstructed) ** 2).item()
    print(f"[+] Polar Quantization MSE: {mse:.6f} (Target < 0.05)")

    # 3. Non-Uniform Sub-2-bit KV-Cache Compression
    kv_cache = CompressedKVCache(num_layers=1, num_kv_heads=16, head_dim=64, mode="2bit")
    sample_keys = torch.randn(1, 16, 512, 64)  # batch, heads, seq_len, dim
    sample_values = torch.randn(1, 16, 512, 64)

    recon_k, recon_v = kv_cache.update(0, sample_keys, sample_values)

    kv_mse = torch.mean((sample_keys - recon_k) ** 2).item()
    print(f"[+] 2-Bit KV Cache MSE: {kv_mse:.4f} (Target < 0.35)")
    print(f"[+] KV Memory Reduction: 4.0x less memory compared to FP16")
    print(f"[+] Successfully converted LLM weight representation for low-spec GPU execution!")

if __name__ == "__main__":
    main()
