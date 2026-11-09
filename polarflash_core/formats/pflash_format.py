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
        if page_alignment <= 0 or (page_alignment & (page_alignment - 1)) != 0:
            raise ValueError(f"page_alignment must be a power of 2, got {page_alignment}")
        self.alignment = page_alignment
        self.tensors: List[Dict[str, Any]] = []
        self._tensor_names: set = set()
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
        Guarantees uniqueness of tensor names and validates payload integrity.
        """
        if not name or not isinstance(name, str):
            raise ValueError("Tensor name must be a non-empty string.")
        if name in self._tensor_names:
            raise ValueError(f"Duplicate tensor name '{name}' registered in .pflash archive.")

        self._tensor_names.add(name)
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
    Provides rigorous checksum validation, file boundary checks, and lifetime tracking.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archive not found: {filepath}")

        file_size = os.path.getsize(filepath)
        if file_size < 16:
            raise ValueError(f"Truncated .pflash file: size is {file_size} bytes (minimum is 16 bytes).")

        self.file_size = file_size
        self.file_obj = open(filepath, "rb")
        self.mm = mmap.mmap(self.file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        self._closed = False

        # Parse preamble
        magic = self.mm[0:4]
        if magic != PFLASH_MAGIC:
            self.close()
            raise ValueError(f"Invalid .pflash magic bytes: {magic}")

        version, alignment, header_len = struct.unpack("<III", self.mm[4:16])
        if version != PFLASH_VERSION:
            self.close()
            raise ValueError(f"Unsupported .pflash format version: {version} (expected {PFLASH_VERSION})")

        if alignment <= 0 or (alignment & (alignment - 1)) != 0:
            self.close()
            raise ValueError(f"Invalid .pflash page alignment: {alignment} (must be a power of 2)")

        if 16 + header_len > file_size:
            self.close()
            raise ValueError(f"Corrupted .pflash file: header length ({header_len}) extends beyond file size ({file_size})")

        self.version = version
        self.alignment = alignment

        # Parse JSON Header safely
        try:
            header_raw = self.mm[16:16 + header_len].decode("utf-8")
            header_data = json.loads(header_raw)
        except Exception as e:
            self.close()
            raise ValueError(f"Malformed .pflash header JSON: {e}") from e

        self.metadata = header_data.get("metadata", {})
        tensors_list = header_data.get("tensors", [])

        self.tensor_catalog: Dict[str, Dict[str, Any]] = {}
        for t in tensors_list:
            t_name = t["name"]
            t_offset = t["offset"]
            t_size = t["size"]

            # Validate tensor boundaries against file size
            if t_offset + t_size > file_size:
                self.close()
                raise ValueError(
                    f"Corrupted .pflash archive: tensor '{t_name}' payload (offset {t_offset}, size {t_size}) "
                    f"exceeds file boundary ({file_size})"
                )

            self.tensor_catalog[t_name] = t

    @property
    def catalog(self) -> Dict[str, Dict[str, Any]]:
        return self.tensor_catalog

    def get_tensor_meta(self, name: str) -> Dict[str, Any]:
        self._ensure_open()
        if name not in self.tensor_catalog:
            raise KeyError(f"Tensor '{name}' not found in .pflash archive")
        return self.tensor_catalog[name]

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError(f"Operation attempted on closed PFlashReader ({self.filepath})")

    def get_tensor_view(self, name: str, verify_checksum: bool = True) -> memoryview:
        """Zero-copy memoryview slice of tensor payload from memory map."""
        self._ensure_open()
        if name not in self.tensor_catalog:
            raise KeyError(f"Tensor '{name}' not found in .pflash archive")

        info = self.tensor_catalog[name]
        offset = info["offset"]
        size = info["size"]

        view = memoryview(self.mm)[offset:offset + size]
        if verify_checksum:
            crc = zlib.crc32(view)
            if crc != info["crc32"]:
                raise ValueError(f"CRC32 mismatch for tensor '{name}': expected {info['crc32']}, got {crc}")

        return view

    def get_tensor_bytes(self, name: str, verify_checksum: bool = True) -> bytes:
        """Returns byte content of tensor payload."""
        view = self.get_tensor_view(name, verify_checksum=verify_checksum)
        return bytes(view)

    def close(self):
        if not self._closed:
            self._closed = True
            if hasattr(self, "mm") and self.mm:
                try:
                    self.mm.close()
                except Exception:
                    pass
            if hasattr(self, "file_obj") and self.file_obj:
                try:
                    self.file_obj.close()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
