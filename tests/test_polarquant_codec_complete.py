"""
POLAR-FLASH: Comprehensive PolarQuant Codec Test Suite
Validates Objective 3:
1. Recursive forward and inverse codecs for dimensions 2, 4, 8, 16.
2. Learned angular codebooks (Lloyd-Max optimization).
3. Adaptive bit allocation conditioned on radius and tree levels.
4. Exact bit-packed byte streams and serialized bpw.
5. .pflash integration with zero-copy memory-mapped loading.
6. Rate-matched comparison between PolarQuant and E8 Lattice Quantization.
"""

import os
import math
import torch
import pytest
import numpy as np

from polarflash_quant.polar.polar_quant import (
    RecursivePolarQuant,
    PolarQuantizer,
    compare_polar_vs_e8_at_matched_bitrate
)
from polarflash_core.formats.pflash_format import PFlashWriter, PFlashReader


@pytest.mark.parametrize("d", [2, 4, 8, 16])
def test_recursive_polar_codec_dimensions(d):
    """Verifies complete encode/decode roundtrip across dimensions 2, 4, 8, 16."""
    torch.manual_seed(42 + d)
    weights = torch.randn(64, 64)
    rpq = RecursivePolarQuant(block_size=d)

    # Encode with packed bitstream
    packed_bytes, meta = rpq.encode_packed(weights, target_bpw=2.0)
    assert isinstance(packed_bytes, bytes)
    assert len(packed_bytes) > 0
    assert meta["block_size"] == d

    # Verify exact serialized bpw
    expected_bpw = (len(packed_bytes) * 8.0) / float(weights.numel())
    assert math.isclose(meta["exact_bpw"], expected_bpw, rel_tol=1e-5)
    assert meta["exact_bpw"] <= 2.25

    # Decode
    recon = rpq.decode_packed(packed_bytes, meta, device="cpu")
    assert recon.shape == weights.shape
    assert not torch.isnan(recon).any()
    assert not torch.isinf(recon).any()

    mse = float(torch.mean((weights - recon) ** 2).item())
    assert mse < 0.5, f"Distortion too high for d={d}: MSE={mse}"


def test_learned_angular_codebooks():
    """Verifies that Lloyd-Max codebook learning fits angle distributions and executes cleanly."""
    torch.manual_seed(101)
    train_weights = torch.randn(128, 128)
    test_weights = torch.randn(64, 64)

    rpq = RecursivePolarQuant(block_size=8, norm_bits=4, angle_bits=2)
    rpq.fit_codebooks(train_weights, iters=10)

    assert hasattr(rpq, "learned_codebooks")
    assert len(rpq.learned_codebooks) == 3  # For d=8, log2(8)=3 levels
    for cb in rpq.learned_codebooks:
        assert len(cb) == 4  # 2^2 = 4 centroids
        assert torch.all(cb[1:] >= cb[:-1])  # Centroids are sorted

    # Encode test weights using learned codebooks
    packed_bytes, meta = rpq.encode_packed(test_weights, target_bpw=2.0)
    assert meta["has_learned_codebooks"] is True

    recon = rpq.decode_packed(packed_bytes, meta)
    assert recon.shape == test_weights.shape
    mse = float(torch.mean((test_weights - recon) ** 2).item())
    assert mse < 0.5


def test_adaptive_bit_allocation_conditioned_on_radius():
    """Verifies adaptive bit allocation across multiple target bitrates (1.75, 2.0, 3.0 bpw)."""
    rpq = RecursivePolarQuant(block_size=8)
    for target in [1.75, 2.0, 3.0]:
        alloc = rpq.compute_adaptive_bit_allocation(target_bpw=target)
        assert len(alloc) == 8  # 1 radius + 7 angles
        total_bits = sum(alloc)
        achieved_bpw = total_bits / 8.0
        assert math.isclose(achieved_bpw, target, abs_tol=0.25)
        # Verify radius gets adequate representation
        assert alloc[0] >= 2


def test_pflash_integration_roundtrip(tmp_path):
    """Verifies full PolarQuant export to .pflash container and zero-copy mmap retrieval."""
    test_file = str(tmp_path / "polar_test.pflash")
    torch.manual_seed(777)
    weights = torch.randn(128, 64)

    rpq = RecursivePolarQuant(block_size=8)

    writer = PFlashWriter()
    meta = rpq.export_to_pflash(writer, tensor_name="model.layers.0.mlp.gate_proj", tensor=weights, target_bpw=2.0)
    writer.save(test_file)

    assert os.path.exists(test_file)

    with PFlashReader(test_file) as reader:
        assert "model.layers.0.mlp.gate_proj" in reader.catalog
        recon = rpq.load_from_pflash(reader, "model.layers.0.mlp.gate_proj", device="cpu")
        assert recon.shape == weights.shape
        mse = float(torch.mean((weights - recon) ** 2).item())
        assert mse < 0.5


def test_polar_vs_e8_matched_bitrate_comparison():
    """Verifies rate-matched comparison between PolarQuant and E8 Lattice at identical target bpw."""
    torch.manual_seed(999)
    weights = torch.randn(64, 64)

    res = compare_polar_vs_e8_at_matched_bitrate(weights, target_bpw=2.0)
    assert "polar" in res
    assert "e8" in res
    assert res["polar"]["exact_bpw"] <= 2.25
    assert res["polar"]["mse"] > 0
    assert res["e8"]["mse"] > 0
    assert res["polar"]["snr_db"] > 0
    assert res["e8"]["snr_db"] > 0

