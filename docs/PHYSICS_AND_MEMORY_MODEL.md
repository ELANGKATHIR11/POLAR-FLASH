# POLAR-FLASH: Physical Memory, Storage & Hardware Roofline Engine

This document specifies the physical storage hierarchy, memory accounting models, NAND flash alignment dynamics, and hardware roofline bounds implemented in POLAR-FLASH.

---

## 1. Physical Memory Accounting

### 1.1 Storage Footprint Accounting
The total on-disk storage footprint of a `.pflash` container consists of all data payloads and structural overhead:

\[
S_{\text{total}} = S_{\text{weights}} + S_{\text{metadata}} + S_{\text{codebooks}} + S_{\text{indices}} + S_{\text{residuals}} + S_{\text{padding}}
\]

where:
- $S_{\text{weights}} = \frac{N \cdot b_{\text{nominal}}}{8}$ bytes.
- $S_{\text{metadata}}$: JSON tensor catalog and architectural configurations.
- $S_{\text{codebooks}}$: Radial and angular floating-point centroid codebooks.
- $S_{\text{indices}}$: Packed bitstreams for polar and lattice coordinates.
- $S_{\text{residuals}}$: Sparse ternary coordinates (sign bits + column indices).
- $S_{\text{padding}}$: Page-alignment padding bytes ensuring $4096$-byte coalescing.

### 1.2 Exact Serialized Effective Bitrate
POLAR-FLASH rejects nominal bit-rate claims that hide codebook and index overheads. The true effective bitrate is strictly computed from total archive size on disk:

\[
b_{\text{effective}} = \frac{8 \cdot S_{\text{file}}}{N_{\text{params}}}
\]

### 1.3 Active Runtime Peak VRAM/RAM
During inference, total active memory is governed by:

\[
M_{\text{peak}} = M_{\text{weights}} + M_{\text{activations}} + M_{\text{KV}} + M_{\text{workspace}} + M_{\text{buffers}} + M_{\text{runtime}}
\]

where:
- $M_{\text{weights}}$: Resident weights in VRAM (or zero under streaming mode).
- $M_{\text{activations}}$: Forward pass intermediate activation tensors ($2 \cdot B \cdot S \cdot d \cdot \text{sizeof}(\text{FP16})$).
- $M_{\text{KV}}$: Long-context Key-Value attention cache.
- $M_{\text{workspace}}$: Scratch memory for fused GEMM and dequantization kernels.
- $M_{\text{buffers}}$: Pinned double-buffering prefetch buffers ($2 \times$ single layer size).
- $M_{\text{runtime}}$: PyTorch / CUDA driver context overhead ($\approx 350$ MB).

---

## 2. Flash-Page Coalescing & Storage Dynamics

### 2.1 NAND Page Geometry & Read Amplification
Modern NVMe SSDs read data in physical NAND flash pages (typically $P = 4096$ bytes, $16384$ bytes, or $64$ KB blocks). For a read request of length $L$ bytes starting at byte offset $o$:

\[
N_{\text{pages}} = \left\lceil \frac{(o \pmod P) + L}{P} \right\rceil
\]

Total transferred bytes from hardware storage:

\[
B_{\text{transferred}} = N_{\text{pages}} \cdot P
\]

From this physical reality, POLAR-FLASH derives two fundamental metrics:

#### Storage Utilization ($U$)
\[
U = \frac{L_{\text{useful}}}{B_{\text{transferred}}} = \frac{L}{N_{\text{pages}} \cdot P} \in (0, 1]
\]

#### Read Amplification ($A$)
\[
A = \frac{B_{\text{transferred}}}{L_{\text{useful}}} = \frac{1}{U} \geq 1.0
\]

When tensor data is not page-aligned (e.g. reading a 200-byte codebook starting at offset 4000), read amplification can spike to $20\times$ ($U < 5\%$). By enforcing $P = 4096$ alignment on every `.pflash` tensor payload, POLAR-FLASH guarantees $U \approx 100\%$ and $A \approx 1.0\times$.

---

## 3. Asynchronous Double-Buffering Pipeline

When weights cannot fit entirely into VRAM, layers must be streamed from NVMe or Host DRAM during token generation. Under naive serial execution:

\[
T_{\text{serial}} = \sum_{l=1}^L \left( T_{\text{compute}}^{(l)} + T_{\text{I/O}}^{(l)} \right)
\]

POLAR-FLASH implements asynchronous double-buffering using dedicated prefetch queues:
- **Buffer A**: Holds active weights for Layer $l$, servicing GPU compute.
- **Buffer B**: Asynchronously streams weights for Layer $l+1$ over PCIe.

The pipelined execution time per layer is:

\[
T_{\text{pipeline}}^{(l)} \approx \max\left( T_{\text{compute}}^{(l)}, T_{\text{I/O}}^{(l)} \right)
\]

If $T_{\text{compute}} \geq T_{\text{I/O}}$, I/O is completely hidden behind computation, achieving a near-theoretical $2\times$ pipeline speedup:

\[
\text{Speedup} = \frac{T_{\text{compute}} + T_{\text{I/O}}}{\max(T_{\text{compute}}, T_{\text{I/O}})}
\]

---

## 4. Hardware Roofline Performance Bounds

The Roofline Model determines the attainable performance $P_{\text{attainable}}$ of a layer as a function of hardware compute capability and operational intensity.

### 4.1 Operational Intensity (OI)
\[
I = \frac{\text{Total Arithmetic Operations (FLOPs)}}{\text{Total Memory Traffic (Bytes Transferred)}} \quad [\text{FLOPs/Byte}]
\]

For a transformer linear projection $Y = X W^T$ with input shape $(B, S, d_{\text{in}})$ and weight shape $(d_{\text{out}}, d_{\text{in}})$:
- Total FLOPs: $2 \cdot B \cdot S \cdot d_{\text{in}} \cdot d_{\text{out}}$
- Transferred Bytes: $d_{\text{in}} d_{\text{out}} \cdot \left(\frac{b}{8}\right) + 2 B S d_{\text{in}} + 2 B S d_{\text{out}}$

### 4.2 Attainable Throughput Bound
\[
P_{\text{attainable}} = \min\left( P_{\text{peak}}, \text{BW}_{\text{memory}} \cdot I \right) \quad [\text{TFLOPS}]
\]

- **Memory-Bandwidth Bound Regime** ($I < I_{\text{knee}}$): Throughput is linearly limited by memory bus speed: $P = \text{BW}_{\text{memory}} \cdot I$.
- **Compute-Bound Regime** ($I \geq I_{\text{knee}}$): Throughput reaches hardware compute saturation $P_{\text{peak}}$.

### 4.3 Why 1.75 bpw Transforms Performance
At batch size $B = 1$ (interactive generation), token decoding is strictly memory-bandwidth bound. Compressing weights from $16.0$ bpw (FP16) down to $1.75$ bpw reduces weight memory traffic by $9.14\times$, increasing operational intensity by over $8\times$ and directly speeding up token generation throughput on bandwidth-constrained hardware.

---

## 5. Physical Energy Consumption Model

The energy dissipated during inference is modeled using component-level physical energy costs:

\[
E_{\text{total}} = N_{\text{op}} \cdot E_{\text{op}} + \sum_{j} B_j \cdot \epsilon_j + N_{\text{IO}} \cdot E_{\text{IO}}
\]

where:
- $E_{\text{op}} \approx 1.0 \text{ pJ}$ per FP16 arithmetic operation.
- $\epsilon_{\text{VRAM}} \approx 5.0 \text{ pJ/byte}$ for GDDR6/HBM bus transfers.
- $\epsilon_{\text{DRAM}} \approx 20.0 \text{ pJ/byte}$ for DDR5 host memory bus transfers.
- $E_{\text{IO}} \approx 100.0 \text{ pJ/byte}$ for PCIe NVMe flash transfers.

Because memory access dissipates $5\times$ to $100\times$ more energy than arithmetic, extreme weight compression is directly an energy-efficiency optimization, drastically reducing laptop battery drain and thermal throttling.

---

## 6. Target Hardware Feasibility Gating

POLAR-FLASH includes automated memory safety verification gates before model instantiation:

| Target Platform | VRAM Budget | Model Size (1.75 bpw) | KV-Cache (8K Tokens) | Workspace & Context | Total Active | Feasibility |
|---|---|---|---|---|---|---|
| **Low-Spec Edge Device** | 4.0 GB | 0.68 GB (3.5B) | 0.50 GB (INT8) | 0.70 GB | 1.88 GB | **FEASIBLE (53% Headroom)** |
| **RTX 5060 Laptop GPU** | 8.0 GB | 0.68 GB (3.5B) | 0.50 GB (INT8) | 0.70 GB | 1.88 GB | **FEASIBLE (76% Headroom)** |
| **Extreme 14B on 8GB VRAM** | 8.0 GB | 3.06 GB (14B) | 1.50 GB (2-Bit) | 1.00 GB | 5.56 GB | **FEASIBLE (30% Headroom)** |

