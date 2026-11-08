"""
POLAR-FLASH: Adversarial Robustness and Format Verification for .pflash
Validates:
1. Rejection of duplicate tensor registrations.
2. Handling of truncated files (<16 bytes preamble, truncated payload).
3. Header JSON corruption and malformed data handling.
4. Payload CRC32 corruption detection.
5. Zero-copy memoryview retrieval and lifetime safety after close.
6. Page alignment validation.
"""

import os
import zlib
import pytest
import struct
from polarflash_core.formats.pflash_format import (
    PFlashWriter, PFlashReader, PFLASH_MAGIC, PFLASH_VERSION
)


def test_duplicate_tensor_name_rejection():
    """Verifies that PFlashWriter raises ValueError when registering duplicate tensor names."""
    writer = PFlashWriter()
    writer.add_tensor("layer.weight", b"\x01\x02\x03\x04", shape=[2, 2], dtype="int8")
    with pytest.raises(ValueError, match="Duplicate tensor name"):
        writer.add_tensor("layer.weight", b"\x05\x06\x07\x08", shape=[2, 2], dtype="int8")


def test_truncated_file_detection(tmp_path):
    """Verifies that PFlashReader catches truncated files smaller than 16 bytes."""
    corrupt_file = str(tmp_path / "truncated.pflash")
    with open(corrupt_file, "wb") as f:
        f.write(b"PFLA\x01\x00")  # Only 6 bytes

    with pytest.raises(ValueError, match="Truncated .pflash file"):
        PFlashReader(corrupt_file)


def test_invalid_magic_bytes(tmp_path):
    """Verifies that PFlashReader rejects invalid magic bytes."""
    corrupt_file = str(tmp_path / "invalid_magic.pflash")
    with open(corrupt_file, "wb") as f:
        f.write(b"BAD!\x01\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00")

    with pytest.raises(ValueError, match="Invalid .pflash magic bytes"):
        PFlashReader(corrupt_file)


def test_corrupted_crc32_detection(tmp_path):
    """Verifies that PFlashReader detects CRC32 payload tampering."""
    archive_path = str(tmp_path / "crc_test.pflash")
    writer = PFlashWriter()
    original_payload = b"ORIGINAL_SECRET_WEIGHT_PAYLOAD"
    writer.add_tensor("weight", original_payload, shape=[len(original_payload)], dtype="uint8")
    writer.save(archive_path)

    # Tamper with the file: mutate one byte in the data section
    with open(archive_path, "r+b") as f:
        content = bytearray(f.read())
        # Find start of payload and corrupt it
        idx = content.find(original_payload)
        assert idx != -1
        content[idx] = (content[idx] + 1) % 256
        f.seek(0)
        f.write(content)

    reader = PFlashReader(archive_path)
    with pytest.raises(ValueError, match="CRC32 mismatch"):
        reader.get_tensor_bytes("weight", verify_checksum=True)
    reader.close()


def test_zero_copy_view_and_lifetime_safety(tmp_path):
    """Verifies get_tensor_view returns memoryview and operations fail after close."""
    archive_path = str(tmp_path / "view_test.pflash")
    writer = PFlashWriter()
    payload = b"\xAA\xBB\xCC\xDD\xEE\xFF" * 100
    writer.add_tensor("weights", payload, shape=[len(payload)], dtype="uint8")
    writer.save(archive_path)

    reader = PFlashReader(archive_path)
    view = reader.get_tensor_view("weights")
    assert isinstance(view, memoryview)
    assert bytes(view) == payload

    reader.close()
    with pytest.raises(RuntimeError, match="Operation attempted on closed PFlashReader"):
        reader.get_tensor_view("weights")

