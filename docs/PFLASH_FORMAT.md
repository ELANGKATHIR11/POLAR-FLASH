# POLAR-FLASH: .pflash Container Format Specification (v1.0)

## 1. Overview & Design Principles

The `.pflash` file format is an open, high-performance, page-aligned binary container designed specifically for extreme neural network compression, low-RAM streaming, and zero-copy memory mapping (`mmap`).

Key Design Objectives:
1. **Zero-Copy Loading**: The operating system maps the file directly into virtual memory pages without requiring deserialization or copying into heap memory.
2. **Flash-Page Alignment**: Every tensor payload starts on a multiple of 4096 bytes (SSD NAND flash page boundary), eliminating storage read amplification and cache line contention.
3. **Exact Bitrate Transparency**: The archive header stores complete parameter counts, bitrates, and metadata, guaranteeing that $b_{\text{effective}} = \frac{8 \cdot S_{\text{file}}}{N}$.
4. **Data Integrity**: Every tensor includes a CRC32 checksum verified at access time.

---

## 2. Binary Layout Specification

```
+-------------------------------------------------------------------------+
|                              PREAMBLE (16 Bytes)                        |
|  [0..3]   : Magic Bytes: b"PFLA" (0x50, 0x46, 0x4C, 0x41)               |
|  [4..7]   : Format Version: uint32 (Little Endian, Current = 1)          |
|  [8..11]  : Page Alignment: uint32 (Little Endian, Default = 4096)      |
|  [12..15] : Header JSON Byte Length (H): uint32 (Little Endian)         |
+-------------------------------------------------------------------------+
|                            HEADER JSON (H Bytes)                        |
|  [16 .. 16+H-1]: UTF-8 Encoded JSON Document containing                 |
|                  - "metadata": Global model configurations              |
|                  - "tensors" : Array of tensor descriptor records        |
+-------------------------------------------------------------------------+
|                         HEADER PADDING (P_0 Bytes)                      |
|  Zero-byte padding ensuring offset(Tensor_0) % Page Alignment == 0      |
+-------------------------------------------------------------------------+
|                          TENSOR PAYLOAD 0                               |
|  [offset_0 .. offset_0 + size_0 - 1]: Packed Byte Stream for Tensor 0   |
+-------------------------------------------------------------------------+
|                         TENSOR 0 PADDING (P_1 Bytes)                    |
|  Zero-byte padding ensuring offset(Tensor_1) % Page Alignment == 0      |
+-------------------------------------------------------------------------+
|                                  ...                                    |
+-------------------------------------------------------------------------+
|                          TENSOR PAYLOAD N-1                             |
|  [offset_{N-1} .. offset_{N-1} + size_{N-1} - 1]: Payload Tensor N-1    |
+-------------------------------------------------------------------------+
```

---

## 3. Header JSON Schema

The JSON header provides structured cataloging of all contained tensors:

```json
{
  "metadata": {
    "source_model": "Qwen3.5-4B-Instruct",
    "backend": "polar",
    "architecture": "Qwen2ForCausalLM",
    "hidden_size": 2560,
    "hadamard_rotation": true,
    "total_params": 3110000000,
    "creation_time": "2026-11-08T22:30:00Z"
  },
  "tensors": [
    {
      "name": "model.layers.0.self_attn.q_proj.weight",
      "shape": [2560, 2560],
      "dtype": "polar",
      "size": 1440000,
      "offset": 4096,
      "crc32": 3184920174,
      "quant_meta": {
        "dtype": "polar",
        "r_bits": 4,
        "theta_bits": 4,
        "r_scale": 0.008412,
        "bpw": 4.0,
        "hadamard_rotation": true,
        "hadamard_block_size": 512,
        "mse": 0.000014,
        "snr_db": 18.42
      }
    }
  ]
}
```

---

## 4. Supported Data Types (`dtype`)

| Identifier | Description | Nominal Bits/Param | Bit-Packing Structure |
|---|---|---|---|
| `"fp16"` | IEEE 754 Half-Precision Float | 16.0 bpw | Standard 16-bit float |
| `"bf16"` | Brain Floating Point | 16.0 bpw | Standard 16-bit bfloat |
| `"int8"` | Symmetric Signed 8-bit Integer | 8.0 bpw | 1 byte per value with affine scale |
| `"int4"` | Symmetric Signed 4-bit Integer | 4.0 bpw | 2 values per byte (nibble packed) |
| `"int3"` | Symmetric Signed 3-bit Integer | 3.0 bpw | 8 values per 3 bytes (24 bits) |
| `"int2"` | Symmetric Signed 2-bit Integer | 2.0 bpw | 4 values per byte |
| `"e8_lattice"` | Gosset $E_8$ Root Lattice VQ | 2.0 bpw | 8D vectors mapped to Shell 1 roots |
| `"e8_rvq"` | Multi-Stage Residual $E_8$ VQ | 3.0 - 4.0 bpw | Cascaded residual root vectors |
| `"polar"` | Recursive PolarQuant | 1.75 - 4.0 bpw | Packed radius and angular bitstream |

---

## 5. Memory-Mapped Access Workflow

```python
from polarflash import PFlashReader

with PFlashReader("model.pflash") as reader:
    # 1. Zero-copy slice directly from OS page cache
    tensor_bytes = reader.get_tensor_bytes("model.layers.0.self_attn.q_proj.weight")
    
    # 2. Dequantize on-the-fly to CPU or GPU
    from polarflash import UnifiedQuantizer
    meta = reader.tensor_catalog["model.layers.0.self_attn.q_proj.weight"]
    w = UnifiedQuantizer.dequantize(tensor_bytes, meta["shape"], meta["dtype"], meta["quant_meta"])
```
