"""
POLAR-FLASH: .pflash Binary Container Format & Memory-Mapped Serialization Engine
Implements Part VI Section 21 of the POLAR-FLASH Specification:
1. Magic bytes b"PFLA" + Format version + Page-aligned Tensor Directory.
2. Zero-copy Memory-Mapped (mmap) tensor retrieval.
3. Storage-aware page alignment (4KB / 16KB / 64KB) for optimal Flash-page coalescing.
4. Full CRC32 checksum validation for numerical integrity.
5. Exact serialized effective bits-per-parameter accounting (b_effective = 8 * S_file / N).
"""

import os
import mmap
import json
import zlib
import struct
import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

# Format Constants
PFLASH_MAGIC = b"PFLA"
PFLASH_VERSION = 1
DEFAULT_PAGE_ALIGNMENT = 4096  # 4KB boundary for SSD/DRAM page-coalescing


class PFlashWriter:
    """
    Serializes quantized model parameters, codebooks, scales, and residuals
    into the optimized, memory-mappable .pflash binary format.
    """
    def __init__(self, page_alignment: int = DEFAULT_PAGE_ALIGNMENT):
        self.alignment = page_alignment
        self.tensors: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def set_metadata(self, meta: Dict[str, Any]):
        self.metadata = meta

    def add_tensor(
        self,
        name: str,
        data_bytes: bytes,
        shape: List[int],
        dtype: str,
        quant_meta: Optional[Dict[str, Any]] = None
    ):
        """
        Registers a raw byte tensor with its quantization metadata.
        """
        crc = zlib.crc32(data_bytes)
        self.tensors.append({
            "name": name,
            "bytes": data_bytes,
            "size": len(data_bytes),
            "shape": shape,
            "dtype": dtype,
            "quant_meta": quant_meta or {},
            "crc32": crc
        })

    def save(self, filepath: str) -> Dict[str, Any]:
        """
        Writes out the .pflash file with page-aligned offsets.
        File Layout:
        [0..3]    Magic Bytes: b"PFLA"
        [4..7]    Version: uint32
        [8..11]   Page Alignment: uint32
        [12..15]  Header JSON Length: uint32
        [16..16+H] Header JSON (metadata + tensor catalog with relative data offsets)
        [Aligned Data Payloads with zero-padding]
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        catalog = []
        payload_bytes_list = []
        current_offset = 0

        # We first estimate header size to align the first tensor payload
        for t in self.tensors:
            catalog.append({
                "name": t["name"],
                "size": t["size"],
                "shape": t["shape"],
                "dtype": t["dtype"],
                "quant_meta": t["quant_meta"],
                "crc32": t["crc32"]
            })
            payload_bytes_list.append(t["bytes"])

        header_dict = {
            "metadata": self.metadata,
            "tensors": catalog
        }

        header_bytes = json.dumps(header_dict).encode("utf-8")
        header_len = len(header_bytes)

        # File preamble: Magic (4) + Version (4) + Alignment (4) + HeaderLen (4) = 16 bytes
        data_start = 16 + header_len
        # Align data_start to self.alignment
        rem = data_start % self.alignment
        pad_header = (self.alignment - rem) if rem != 0 else 0
        data_start_aligned = data_start + pad_header

        # Now compute exact aligned file offsets for each tensor payload
        cur_payload_offset = data_start_aligned
        for entry, pbytes in zip(catalog, payload_bytes_list):
            entry["offset"] = cur_payload_offset
            # Align next tensor to page boundary
            tensor_len = len(pbytes)
            t_rem = tensor_len % self.alignment
            t_pad = (self.alignment - t_rem) if t_rem != 0 else 0
            cur_payload_offset += (tensor_len + t_pad)

        # Re-encode header with exact absolute file offsets
        header_dict["tensors"] = catalog
        header_bytes = json.dumps(header_dict).encode("utf-8")
        pad_header = data_start_aligned - (16 + len(header_bytes))

        # Write file sequentially
        with open(filepath, "wb") as f:
            f.write(PFLASH_MAGIC)
            f.write(struct.pack("<III", PFLASH_VERSION, self.alignment, len(header_bytes)))
            f.write(header_bytes)
            if pad_header > 0:
                f.write(b"\x00" * pad_header)

            for entry, pbytes in zip(catalog, payload_bytes_list):
                f.write(pbytes)
                t_pad = (entry["offset"] + entry["size"])
                cur_pos = f.tell()
                pad_needed = ((cur_pos + self.alignment - 1) // self.alignment) * self.alignment - cur_pos
                if pad_needed > 0:
                    f.write(b"\x00" * pad_needed)

        total_file_size = os.path.getsize(filepath)
        total_params = self.metadata.get("total_params", sum(int(np.prod(e["shape"])) for e in catalog))
        effective_bpw = (8.0 * total_file_size) / max(total_params, 1)

        return {
            "filepath": filepath,
            "total_bytes": total_file_size,
            "num_tensors": len(catalog),
            "total_params": total_params,
            "effective_bpw": effective_bpw,
            "page_alignment": self.alignment
        }


class PFlashReader:
    """
    Zero-copy Memory-Mapped (mmap) reader for .pflash archives.
    Allows streaming tensor reads without loading the full model into RAM.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_obj = open(filepath, "rb")
        self.mm = mmap.mmap(self.file_obj.fileno(), 0, access=mmap.ACCESS_READ)

        # Parse preamble
        magic = self.mm[0:4]
        if magic != PFLASH_MAGIC:
            raise ValueError(f"Invalid .pflash magic bytes: {magic}")

        version, alignment, header_len = struct.unpack("<III", self.mm[4:16])
        self.version = version
        self.alignment = alignment

        # Parse JSON Header
        header_raw = self.mm[16:16 + header_len].decode("utf-8")
        header_data = json.loads(header_raw)
        self.metadata = header_data.get("metadata", {})
        self.tensor_catalog: Dict[str, Dict[str, Any]] = {
            t["name"]: t for t in header_data.get("tensors", [])
        }

    def get_tensor_bytes(self, name: str, verify_checksum: bool = True) -> bytes:
        """Zero-copy slice of tensor payload from memory map."""
        if name not in self.tensor_catalog:
            raise KeyError(f"Tensor '{name}' not found in .pflash archive")

        info = self.tensor_catalog[name]
        offset = info["offset"]
        size = info["size"]

        raw_bytes = self.mm[offset:offset + size]
        if verify_checksum:
            crc = zlib.crc32(raw_bytes)
            if crc != info["crc32"]:
                raise ValueError(f"CRC32 mismatch for tensor {name}: expected {info['crc32']}, got {crc}")

        return raw_bytes

    def close(self):
        if self.mm:
            self.mm.close()
        if self.file_obj:
            self.file_obj.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
