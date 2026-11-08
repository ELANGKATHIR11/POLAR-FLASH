# POLAR-FLASH Framework Architecture

## 1. High-Level System Architecture

POLAR-FLASH is a mathematically grounded, physics-aware framework engineered for extreme Large Language Model (LLM) weight compression, sub-2-bit quantization, zero-copy flash streaming, structured activation sparsity, quantized KV-cache compression, and compressed-domain parameter-efficient fine-tuning.

```
+-----------------------------------------------------------------------------------+
|                              MODEL INGESTION & CODEC                              |
|   Hugging Face Safetensors / PyTorch Checkpoints (Llama, Qwen, Mistral, Vision)   |
|                                         |                                         |
|                  +----------------------+----------------------+                  |
|                  | Block Randomized Walsh-Hadamard Transform   |                  |
|                  | (Disperses outlier coordinate spikes)       |                  |
|                  +----------------------+----------------------+                  |
|                                         |                                         |
|    +-------------------+----------------+-------------------+----------------+    |
|    |                   |                                    |                |    |
|    v                   v                                    v                v    |
|  Recursive Polar     E8 Root Lattice                      INT4 / INT3      FP16 / |
|  Quant (Tree Codec)  RVQ Multi-Stage                      INT2 Packing     BF16   |
+----+-------------------+----------------+-------------------+----------------+----+
     |                   |                |                   |                |
     +-------------------+----------------+-------------------+----------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        .pflash CONTAINER SERIALIZATION                            |
|  - Magic Header [b"PFLA"] + Format Version v1                                     |
|  - Storage-Aligned Tensor Catalog (4KB / 16KB / 64KB Page Boundaries)             |
|  - Exact Bit-Packed Payloads (b_eff = 8 * S_file / N)                             |
|  - CRC32 Numerical Integrity Checksums                                            |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            HARDWARE AutoEngine FACTORY                            |
|  - Hardware Introspection: CPU SIMD (AVX2/AVX-512), Host RAM, GPU VRAM, SSD BW    |
|  - Thermal Governor: Enforces duty-cycle safety for laptops (<65C threshold)      |
|  - Execution Dispatcher: CPU Engine vs CUDA Resident vs CUDA Streaming Pipeline  |
+-----------------------------------------+-----------------------------------------+
                                          |
          +-------------------------------+-------------------------------+
          |                                                               |
          v                                                               v
+-----------------------------------+           +-----------------------------------+
|        PFlashCPUEngine            |           |         PFlashCUDAEngine          |
|  - Zero-Copy mmap Slices          |           |  - Asynchronous CUDA Streams      |
|  - AVX2 Vectorized Dequantization |           |  - Pinned Double-Buffering        |
|  - Fused Int4/Int2 GEMV Kernels   |           |  - Tensor Core MatMul Execution   |
+-----------------+-----------------+           +-----------------+-----------------+
                  |                                               |
                  +-----------------------+-----------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                              PHYSICAL RUNTIME SUBSYSTEMS                          |
|  1. Flash Streamer: Double-buffering prefetch (T_pipe ~ max(T_comp, T_io))        |
|  2. CompressedKVCache: GQA Cache in FP16, INT8, 2-Bit Asym, and Polar-QJL         |
|  3. ContextualSparseFFN: Top-K SwiGLU activation gather (50-70% FLOPs reduction)  |
|  4. LookaheadNeuronRouter: Predictive Layer L+1 neuron selection                  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      COMPRESSED-DOMAIN FINE-TUNING ENGINES                        |
|  1. PV-GradCode: Alternating Continuous V-Step (AdamW) + Discrete P-Step          |
|  2. PolarLoRALinear: Frozen quantized base layer + rank-r trainable adapters      |
|  3. PolarQATLinear: Straight-Through Estimator (STE) differentiable quantization  |
|  4. TIESWeightMerger: Trim, Elect Sign, & Merge for multi-task domain adapters     |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Subsystems & Components

### 2.1 Ingestion & Unified Quantization Engine (`polarflash_core`, `polarflash_quant`)
- **`ArchitectureAdapter`**: Standardizes tensor naming conventions across Llama, Qwen, Mistral, and multimodal vision projectors (e.g. `c_attn`, `q_proj`, `w1`, `merger`).
- **`UnifiedQuantizer`**: Universal interface implementing 8 distinct quantization backends:
  - `fp16` & `bf16`: Full precision references.
  - `int8`: Symmetric 8-bit integer with per-tensor/per-channel scale.
  - `int4`: Packed 4-bit representation (2 nibbles per byte, 4.0 nominal bpw).
  - `int3`: Packed 3-bit representation (8 values per 3 bytes, 3.0 nominal bpw).
  - `int2`: Packed 2-bit representation (4 values per byte, 2.0 nominal bpw).
  - `e8`: Gosset $E_8$ lattice vector quantization with Shell 1 root codebooks.
  - `polar`: Recursive PolarQuant with radial and angular bit packing (sub-2.0 nominal bpw).
- **`BlockHadamardTransform`**: High-performance, energy-preserving orthogonal rotation for arbitrary matrix dimensions, suppressing extreme activation/weight outlier spikes before quantization.

### 2.2 Storage & Memory-Mapped Engine (`polarflash_core/formats`)
- **`PFlashWriter`**: Serializes quantized tensor blocks with NAND flash page alignment (4096 bytes), CRC32 validation, and JSON metadata catalog.
- **`PFlashReader`**: Zero-copy reader using Python `mmap`, mapping the archive directly into process virtual memory space without loading multi-gigabyte models into system RAM.

### 2.3 Native Inference Runtime (`polarflash_rt`)
- **`AutoEngine` & `HardwareProbe`**: Discovers platform CPU SIMD instruction sets, host RAM, GPU architecture, VRAM capacity, and storage throughput.
- **`ThermalGovernor`**: Enforces duty-cycle pacing and cooldown delays to prevent thermal shutdowns on mobile/laptop GPUs (such as RTX 5060 Laptop).
- **`PFlashCPUEngine`**: Memory-mapped CPU execution with vectorized dequantization and fused GEMV kernels.
- **`PFlashCUDAEngine`**: High-throughput GPU execution with pinned host memory buffers and asynchronous CUDA stream overlapping.

### 2.4 Physical Optimization Subsystems
- **`DoubleBufferPipeline` & `AsyncDoubleBufferQueue`**: Asynchronously prefetches the weights for Layer $L+1$ while computing Layer $L$, achieving $T_{\text{pipeline}} \approx \max(T_{\text{compute}}, T_{\text{I/O}})$.
- **`CompressedKVCache`**: Memory-saving Key-Value cache engine supporting FP16, INT8 dynamic scaling, 2-Bit uniform, 2-Bit asymmetric, and sliding-window attention eviction with sink tokens.
- **`ContextualSparseFFN`**: Context-dependent SwiGLU activation pruning that selects the top $k$ neurons by magnitude $|a_j(x)|$ and executes sub-matrix gathers on $W_{\text{down}}$, saving up to 50-70% of MLP arithmetic operations with negligible logit distortion.

### 2.5 Fine-Tuning Engine (`polarflash_train`)
- **`PVGradCodeOptimizer`**: Dual-phase alternating optimizer:
  - **V-Step**: Continuous parameter updates (scales $\theta$) using AdamW while discrete codes are frozen.
  - **P-Step**: Discrete code transitions proposed using first-order sensitivity $\Delta L \approx \nabla_c L^T \Delta c$, bounded by a trust region and gated by a rollback verification check.
- **`PolarLoRALinear`**: Freezes the underlying quantized weights $W_{\text{base}}$ and attaches rank-$r$ low-rank adapters ($B \times A$), enabling instruction tuning on consumer GPUs with under 4GB VRAM.
- **`TIESWeightMerger`**: Resolves interference among specialized task vectors using trimming, sign election, and disjoint consensus averaging.
