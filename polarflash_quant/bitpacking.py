"""
POLAR-FLASH: High-Throughput Bit-Packing & Unpacking Codec
Supports exact low-bit representations without zero-padding wastage:
1. INT8: 8.0 bpw (signed int8)
2. INT4: 4.0 bpw (2 signed/unsigned 4-bit nibbles per byte)
3. INT3: 3.0 bpw (8 3-bit values packed into 3 bytes = 24 bits)
4. INT2: 2.0 bpw (4 2-bit values per byte)
5. INT1: 1.0 bpw (8 binary signs per byte)
"""

import torch
import numpy as np
from typing import Tuple, Optional


def pack_int4(tensor_int4: torch.Tensor) -> bytes:
    """
    Packs 4-bit values in range [-8, 7] into uint8 bytes (2 nibbles per byte).
    Last dimension or flattened tensor is padded to even length if necessary.
    """
    flat = tensor_int4.detach().cpu().flatten()
    n = flat.numel()
    if n % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype)])

    # Map signed [-8, 7] to unsigned [0, 15]
    u4 = (flat + 8).to(torch.uint8) & 0x0F
    low = u4[0::2]
    high = u4[1::2]
    packed = low | (high << 4)
    return packed.numpy().tobytes()


def unpack_int4(packed_bytes: bytes, num_elements: int, device: str = "cpu") -> torch.Tensor:
    """
    Unpacks uint8 bytes into signed int8 values in range [-8, 7].
    """
    arr = np.frombuffer(packed_bytes, dtype=np.uint8).copy()
    packed = torch.from_numpy(arr).to(device)
    low = (packed & 0x0F).to(torch.int8) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8

    unpacked = torch.empty(len(packed) * 2, dtype=torch.int8, device=device)
    unpacked[0::2] = low
    unpacked[1::2] = high
    return unpacked[:num_elements]


def pack_int2(tensor_int2: torch.Tensor) -> bytes:
    """
    Packs 2-bit values in range [0, 3] into uint8 bytes (4 values per byte).
    """
    flat = tensor_int2.detach().cpu().flatten()
    n = flat.numel()
    rem = n % 4
    if rem != 0:
        flat = torch.cat([flat, torch.zeros(4 - rem, dtype=flat.dtype)])

    u2 = flat.to(torch.uint8) & 0x03
    v0 = u2[0::4]
    v1 = u2[1::4]
    v2 = u2[2::4]
    v3 = u2[3::4]
    packed = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
    return packed.numpy().tobytes()


def unpack_int2(packed_bytes: bytes, num_elements: int, device: str = "cpu") -> torch.Tensor:
    """
    Unpacks uint8 bytes into unsigned int8 values in range [0, 3].
    """
    arr = np.frombuffer(packed_bytes, dtype=np.uint8).copy()
    packed = torch.from_numpy(arr).to(device)

    v0 = (packed & 0x03).to(torch.int8)
    v1 = ((packed >> 2) & 0x03).to(torch.int8)
    v2 = ((packed >> 4) & 0x03).to(torch.int8)
    v3 = ((packed >> 6) & 0x03).to(torch.int8)

    unpacked = torch.empty(len(packed) * 4, dtype=torch.int8, device=device)
    unpacked[0::4] = v0
    unpacked[1::4] = v1
    unpacked[2::4] = v2
    unpacked[3::4] = v3
    return unpacked[:num_elements]


def pack_int3(tensor_int3: torch.Tensor) -> bytes:
    """
    Packs 3-bit values in range [0, 7] into uint8 bytes (8 values per 3 bytes = 24 bits).
    """
    flat = tensor_int3.detach().cpu().flatten()
    n = flat.numel()
    rem = n % 8
    if rem != 0:
        flat = torch.cat([flat, torch.zeros(8 - rem, dtype=flat.dtype)])

    u3 = flat.to(torch.uint8) & 0x07
    num_groups = len(u3) // 8

    # Extract the 8 lanes
    v0 = u3[0::8].numpy().astype(np.uint32)
    v1 = u3[1::8].numpy().astype(np.uint32)
    v2 = u3[2::8].numpy().astype(np.uint32)
    v3 = u3[3::8].numpy().astype(np.uint32)
    v4 = u3[4::8].numpy().astype(np.uint32)
    v5 = u3[5::8].numpy().astype(np.uint32)
    v6 = u3[6::8].numpy().astype(np.uint32)
    v7 = u3[7::8].numpy().astype(np.uint32)

    # Byte 0: v0 (bits 0..2) | v1 (bits 3..5) | v2_low (bits 6..7)
    b0 = (v0 | (v1 << 3) | ((v2 & 0x03) << 6)).astype(np.uint8)
    # Byte 1: v2_high (bit 0) | v3 (bits 1..3) | v4 (bits 4..6) | v5_low (bit 7)
    b1 = (((v2 >> 2) & 0x01) | (v3 << 1) | (v4 << 4) | ((v5 & 0x01) << 7)).astype(np.uint8)
    # Byte 2: v5_high (bits 0..1) | v6 (bits 2..4) | v7 (bits 5..7)
    b2 = (((v5 >> 1) & 0x03) | (v6 << 2) | (v7 << 5)).astype(np.uint8)

    # Interleave b0, b1, b2
    out = np.empty(num_groups * 3, dtype=np.uint8)
    out[0::3] = b0
    out[1::3] = b1
    out[2::3] = b2
    return out.tobytes()


def unpack_int3(packed_bytes: bytes, num_elements: int, device: str = "cpu") -> torch.Tensor:
    """
    Unpacks 3 bytes into 8 3-bit values in range [0, 7].
    """
    arr = np.frombuffer(packed_bytes, dtype=np.uint8)
    num_groups = len(arr) // 3
    b0 = arr[0::3].astype(np.uint32)
    b1 = arr[1::3].astype(np.uint32)
    b2 = arr[2::3].astype(np.uint32)

    v0 = (b0 & 0x07).astype(np.int8)
    v1 = ((b0 >> 3) & 0x07).astype(np.int8)
    v2 = (((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)).astype(np.int8)
    v3 = ((b1 >> 1) & 0x07).astype(np.int8)
    v4 = ((b1 >> 4) & 0x07).astype(np.int8)
    v5 = (((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)).astype(np.int8)
    v6 = ((b2 >> 2) & 0x07).astype(np.int8)
    v7 = ((b2 >> 5) & 0x07).astype(np.int8)

    unpacked = np.empty(num_groups * 8, dtype=np.int8)
    unpacked[0::8] = v0
    unpacked[1::8] = v1
    unpacked[2::8] = v2
    unpacked[3::8] = v3
    unpacked[4::8] = v4
    unpacked[5::8] = v5
    unpacked[6::8] = v6
    unpacked[7::8] = v7

    t = torch.from_numpy(unpacked[:num_elements]).to(device)
    return t
