# POLAR-FLASH: Engineering & Scientific Research Roadmap

This document outlines the phased research milestones, architectural goals, and production roadmap for the POLAR-FLASH project.

---

## 1. Research Targets & Scientific Milestones

### Milestone 1: The 1.75 Effective bpw Boundary (Current Status: In Progress)
- **Goal**: Rigorously validate the practical usability of 1.75 effective bits per parameter on models ranging from 3B to 70B parameters without severe perplexity degradation.
- **Components**:
  - Recursive PolarQuant coordinate tree optimization.
  - Lloyd-Max non-uniform angular codebook learning.
  - 5% Fisher-weighted ternary residual correction.
  - Post-quantization PV-GradCode fine-tuning to recover instruction-following benchmarks.

### Milestone 2: Native Fused CUDA & Triton Sub-2-bit GEMM Kernels
- **Goal**: Implement native GPU matrix multiplication that decodes packed sub-2-bit weights directly into SRAM / Tensor Core registers on-the-fly.
- **Components**:
  - Custom Triton kernel for 2-bit uniform and 4-bit packed GEMM.
  - Fused dequantize-and-SwiGLU kernel combining $W_{\text{gate}}$ and $W_{\text{up}}$ in a single pass.
  - Shared memory caching for Hadamard sign vectors.

### Milestone 3: Extreme Multi-Modal Vision-Language Streaming
- **Goal**: Enable full vision-language models (e.g. Qwen2-VL, Zenith-4B Vision) to stream high-resolution vision embeddings alongside compressed language backbones.
- **Components**:
  - Dedicated architecture adapters for ViT patch projectors.
  - Dynamic resolution caching to prevent VRAM spikes during high-resolution document OCR.

### Milestone 4: Cross-Platform Acceleration (Apple Silicon & Mobile)
- **Goal**: Expand `polarflash-rt` beyond x86_64 AVX2 and NVIDIA CUDA to ARM NEON, Apple Silicon Metal (MPS), and mobile edge NPU environments.
- **Components**:
  - Apple Metal Performance Shaders (MPS) compute shaders for packed weight decoding.
  - NEON SIMD intrinsics for ARMv8/ARMv9 architectures.

---

## 2. Release & Versioning Schedule

```
+-----------------------------------------------------------------------------+
|                               RELEASE PHASES                                |
+-----------------------------------------------------------------------------+
| v1.0.0 (Current) : Complete Mathematical Foundations & Unified Codec        |
|                    - 8 Quantization Backends (FP16, BF16, INT8, INT4,       |
|                      INT3, INT2, E8 Lattice, PolarQuant)                    |
|                    - .pflash Container Specification & Zero-Copy mmap       |
|                    - Hardware AutoEngine & Thermal Safety Governor          |
|                    - Contextual Sparse FFN & Asynchronous Double-Buffering  |
|                    - 41/41 Verified Automated Unit & Physics Tests          |
+-----------------------------------------------------------------------------+
| v1.1.0 (Q1 2027) : Native Tensor Core Kernels & Triton Integration         |
|                    - Fused Triton Sub-2-bit GEMM for Ampere/Ada/Blackwell   |
|                    - Dynamic KV-Cache Eviction Auto-Tuning                  |
+-----------------------------------------------------------------------------+
| v1.2.0 (Q2 2027) : Multi-Modal Vision & Distributed Pipeline Parallelism    |
|                    - Vision Tower Streamer & Patch Compression              |
|                    - Dual-GPU Double-Buffered NVLink Streaming              |
+-----------------------------------------------------------------------------+
```

