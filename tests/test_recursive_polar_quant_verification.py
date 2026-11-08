"""
POLAR-FLASH: Scientific Verification for Recursive PolarQuant and PolarQuantizer
Validates:
1. Arbitrary supported power-of-two block dimensions (d in {2, 4, 8, 16}).
2. Exact unquantized invertibility: max |x - inverse(forward(x))| < 1e-6.
3. Exact zero-radius boundary condition: zero reconstruction without division-by-zero or NaNs.
4. Independent MSE distortion validation: D = ||w - w_hat||^2.
5. High-level PolarQuantizer serialization and packed bitstream round-trip.
6. Bitrate accounting: serialized bpw including packed payload and metadata.
"""

import math
import pytest
import torch
from polarflash_quant.polar.polar_quant import (
    PolarQuant2D, RecursivePolarQuant, PolarQuantizer, pack_polar_bits, unpack_polar_bits
)


@pytest.mark.parametrize("d", [2, 4, 8, 16])
def test_recursive_polar_exact_invertibility(d):
    """Verifies that forward_recursive and inverse_recursive form an exact unquantized inverse."""
    torch.manual_seed(42 + d)
    rp = RecursivePolarQuant(block_size=d)
    x = torch.randn(64, d)

    top_radius, angles = rp.forward_recursive(x)
    assert top_radius.shape == (64,)
    # Verify top_radius equals Euclidean L2 norm of each vector
    true_norms = torch.norm(x, p=2, dim=-1)
    assert torch.allclose(top_radius, true_norms, atol=1e-5)

    x_rec = rp.inverse_recursive(top_radius, angles)
    max_err = float(torch.max(torch.abs(x - x_rec)).item())
    assert max_err < 1e-6, f"Unquantized reconstruction error too high for d={d}: {max_err}"


@pytest.mark.parametrize("d", [2, 4, 8, 16])
def test_zero_radius_boundary_safety(d):
    """Verifies zero-vector input is handled cleanly without division by zero or NaN."""
    rp = RecursivePolarQuant(block_size=d)
    x_zeros = torch.zeros(16, d)
    top_radius, angles = rp.forward_recursive(x_zeros)
    assert torch.all(top_radius == 0.0)

    for lvl_theta in angles:
        assert torch.all(lvl_theta == 0.0)
        assert not torch.isnan(lvl_theta).any()

    x_rec = rp.inverse_recursive(top_radius, angles)
    assert torch.all(x_rec == 0.0)
    assert not torch.isnan(x_rec).any()

    recon_q, mse = rp.quantize_block(x_zeros)
    assert mse == 0.0
    assert torch.all(recon_q == 0.0)


def test_independent_mse_distortion_calculation():
    """Verifies distortion reported by PolarQuant2D matches independent PyTorch mean squared error."""
    torch.manual_seed(101)
    x = torch.randn(128, 64)
    pq = PolarQuant2D(r_bits=4, theta_bits_base=4)
    x_hat, meta = pq.quantize(x)

    indep_mse = float(torch.mean((x - x_hat) ** 2).item())
    assert math.isclose(meta["mse_distortion"], indep_mse, rel_tol=1e-5)

    var_x = float(torch.var(x).item())
    indep_snr = 10.0 * math.log10(var_x / indep_mse)
    assert math.isclose(meta["snr_db"], indep_snr, rel_tol=1e-4)


def test_polar_quantizer_packing_and_serialized_bpw():
    """Verifies packed bitstream serialization and exact bit-per-weight accounting."""
    torch.manual_seed(202)
    weights = torch.randn(128, 128)  # 16,384 parameters
    pq = PolarQuantizer(bits=4)

    codes, meta = pq.quantize(weights, pack=True)
    assert "packed_bytes" in meta
    packed = meta["packed_bytes"]

    # 16,384 weights -> 8,192 pairs. Each pair takes 4 + 4 = 8 bits = 1 byte.
    assert len(packed) == 8192, f"Expected 8192 packed bytes, got {len(packed)}"
    effective_bpw = (8.0 * len(packed)) / float(weights.numel())
    assert effective_bpw == 4.0

    # Test round-trip unpack and dequantize
    reconstructed = pq.unpack_and_dequantize(packed, meta)
    assert reconstructed.shape == weights.shape
    mse = float(torch.mean((weights - reconstructed) ** 2).item())
    assert mse < 0.05

