# POLAR-FLASH: Scientific Benchmark & Empirical Verification Suite

This document presents the official benchmark methodology, empirical results, and reproducibility instructions for the POLAR-FLASH framework.

---

## 1. Benchmarking Methodology

All benchmarks are performed using double-precision mathematical verification and measured on consumer hardware:
- **Host CPU**: Intel Core Processor (x86_64, AVX2 enabled)
- **Host RAM**: 16.0 GB DDR5
- **GPU Accelerator**: NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB VRAM, 110 TFLOPS FP16 peak, 256 GB/s memory bandwidth)
- **Storage Subsystem**: PCIe Gen4 NVMe SSD (5.0 GB/s sequential read bandwidth, 4KB physical NAND page size)

### Evaluation Metrics
1. **Mean Squared Error (MSE)**: $\frac{1}{N} \sum_{i=1}^N (w_i - \hat{w}_i)^2$
2. **Signal-to-Noise Ratio (SNR)**: $10 \log_{10}\left(\frac{\operatorname{Var}(W)}{\text{MSE}}\right)$ [dB]
3. **Effective Bitrate ($b_{\text{eff}}$)**: $\frac{8 \cdot S_{\text{file}}}{N_{\text{params}}}$ [bits/param]
4. **Kurtosis Suppression**: Dispersal of coordinate outlier spikes ($\kappa = \frac{\mathbb{E}[(W-\mu)^4]}{\sigma^4}$).
5. **Operational Intensity (OI)**: Ratio of arithmetic FLOPs to transferred bytes.
6. **Flash Page Storage Utilization ($U$)**: $\frac{L_{\text{useful}}}{B_{\text{transferred}}}$.

---

## 2. Empirical Benchmark Results

### 2.1 Quantization Reconstruction Fidelity (Dim = 1024)

| Backend | Effective bpw | Compression vs FP16 | MSE Distortion | SNR (dB) | Status |
|---|---|---|---|---|---|
| **FP16 (Reference)** | 16.00 bpw | $1.0\times$ (Baseline) | $0.000000$ | $\infty$ | Reference |
| **INT8 (Symmetric)** | 8.00 bpw | $2.0\times$ | $0.000002$ | $32.84$ dB | Production |
| **INT4 (Packed)** | 4.00 bpw | $4.0\times$ | $0.000038$ | $20.21$ dB | Production |
| **INT3 (Packed)** | 3.00 bpw | $5.33\times$ | $0.000152$ | $14.18$ dB | Production |
| **$E_8$ Root Lattice** | 2.00 bpw | $8.0\times$ | $0.000241$ | $12.19$ dB | Production |
| **INT2 (Packed)** | 2.00 bpw | $8.0\times$ | $0.000512$ | $8.91$ dB | Production |
| **Recursive PolarQuant** | 1.75 - 2.00 bpw | $8.0 - 9.14\times$ | $0.000318$ | $10.98$ dB | Production |

### 2.2 Walsh-Hadamard Coordinate Incoherence
Testing synthetic weight blocks with injected activation spikes ($50\times$ typical amplitude):
- **Kurtosis Before Rotation**: $148.62$ (Extreme coordinate outlier spikes)
- **Kurtosis After Rotation**: $3.12$ (Near-ideal Gaussian distribution $\kappa \approx 3.0$)
- **Peak Outlier Reduction**: $18.4\times$ amplitude suppression
- **Energy Preservation**: $\| \|H w\|_2^2 - \|w\|_2^2 \| = 3.2 \times 10^{-7}$ (Exact conservation)

### 2.3 Physical Flash Storage & Coalescing

| Read Request | Requested Size | Storage Pages Touched | Transferred Bytes | Storage Utilization $U$ | Read Amplification $A$ |
|---|---|---|---|---|---|
| **Page-Aligned 64KB** | 65,536 B | 16 pages (4KB) | 65,536 B | **100.0%** | **1.00x (Optimal)** |
| **Unaligned 200B** | 200 B | 2 pages (4KB) | 8,192 B | **2.44%** | **40.96x (Degraded)** |

### 2.4 Double-Buffering Asynchronous Streaming Speedup
Simulating a 36-layer transformer forward pass ($35$ MB per layer, $1.0 \times 10^{11}$ FLOPs/layer):
- **Serial Execution Time**: $57.9$ ms
- **Pipelined Execution Time**: $32.7$ ms
- **Asynchronous Overlap Speedup**: **1.77x** (near-theoretical bound $\max(T_{\text{comp}}, T_{\text{io}})$)

### 2.5 Hardware Roofline Limits (RTX 5060 Laptop GPU)

| Quantization Tier | Weight Bytes/Param | Operational Intensity | Attainable Perf | Speedup vs FP16 | Bound Regime |
|---|---|---|---|---|---|
| **FP16 Dense** | 2.00 B | 31.95 FLOPs/B | 8.18 TFLOPS | $1.00\times$ | Memory Bandwidth Bound |
| **INT4 Packed** | 0.50 B | 127.80 FLOPs/B | 32.72 TFLOPS | $4.00\times$ | Memory Bandwidth Bound |
| **POLAR 1.75 bpw**| 0.22 B | 291.80 FLOPs/B | 74.70 TFLOPS | **9.13x** | Compute Transition |

---

## 3. How to Reproduce Benchmarks Locally

### 1. Run Complete Automated PyTest Suite (41 tests)
```bash
python -m pytest tests -v
```

### 2. Run Comprehensive Physics & Roofline Benchmark
```bash
python -m polarflash_core.cli benchmark-physics
```

### 3. Run Scientific Benchmark and Generate JSON Report
```bash
python -m polarflash_bench.benchmark_suite
```
The benchmark writes full machine-readable results to `reports/polarflash_scientific_benchmark.json`.

