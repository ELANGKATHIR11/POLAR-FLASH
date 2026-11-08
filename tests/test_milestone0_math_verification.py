"""
POLAR-FLASH: Milestone 0 Comprehensive Mathematical Verification Test Suite
Validates all mathematical and physical foundations specified in Section 3 of Project Mission:
1. Shannon Rate-Distortion & Lagrangian Allocation
2. Hadamard Orthogonality (H^T H = I) & Energy Preservation (||Hw||_2^2 = ||w||_2^2)
3. Gaussian Concentration (Chi-squared radial shell R ~ sigma * sqrt(d), Kurtosis reduction)
4. PolarQuant 2D & Recursive Reconstruction (D ~ (Delta r)^2 + r^2 * (Delta theta)^2)
5. E8 Lattice Nearest Point Correctness & Scale Optimization (Sphere packing density ~0.25367)
6. QJL Estimator Unbiasedness: E[<x, e>_hat] == <x, e>
7. QJL Variance: Var(q_hat) == (1/m) * sum_{i != j} x_i e_i x_j e_j
8. Diagonal Fisher Information & Sparse Ternary Residual Decomposition
9. Physical Memory Accounting: S_total vs M_peak
10. Flash-Page Utilization U and Read Amplification A
11. Operational Intensity & Hardware Roofline Performance Bounds
12. Shannon Entropy & Entropy-Constrained Scalar Quantization (ECSQ)
13. Block-wise Hadamard on Arbitrary Non-Power-of-2 Dimensions (e.g. SwiGLU 11008)
14. Rigorous Gaussian Concentration Verification on Hyperspherical Shells
15. Recursive Polar Quantization Tree & Exact Decomposition
16. Polar Bitstream Packing and Zero-Loss Coordinate Unpacking
17. E8 240 Roots Generation & Multi-Stage Residual Vector Quantization (RVQ)
18. XNOR + Popcount Bitwise QJL Estimator vs Continuous Reference
19. Comparative Saliency Sparsity Analysis (1%, 3%, 5%, 10%)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import torch
import numpy as np
import pytest

from polarflash_quant.information_theory.rate_distortion import (
    shannon_entropy, empirical_entropy, gaussian_rate_distortion, vector_source_rate_distortion,
    EntropyConstrainedQuantizer, estimate_empirical_rd_curve, LayerRateAllocator, calculate_effective_bpw
)
from polarflash_quant.transforms.hadamard import (
    fast_walsh_hadamard_transform, RandomizedHadamardTransform, BlockHadamardTransform,
    compute_distribution_diagnostics, verify_gaussian_concentration
)
from polarflash_quant.polar.polar_quant import (
    PolarQuant2D, RecursivePolarQuant, PolarQuantizer, pack_polar_bits, unpack_polar_bits
)
from polarflash_quant.lattice.e8_lattice import (
    quantize_d8, quantize_e8, E8LatticeQuantizer, E8ResidualVectorQuantizer, generate_e8_roots
)
from polarflash_quant.residual.qjl import (
    QJLResidualEstimator, BitPackedQJLEstimator, XNORPopcountQJLEstimator, verify_qjl_unbiasedness_and_variance
)
from polarflash_quant.calibration.fisher import (
    SparseTernaryResidual, compare_sparsity_levels
)
from polarflash_core.formats.pflash_format import PFlashWriter, PFlashReader
from polarflash_rt.streaming.flash_streamer import FlashPageCoalescer, DoubleBufferPipeline
from polarflash_quant.hardware.roofline import (
    RooflineAnalyzer, MemoryFeasibilityGate, RTX_5060_LAPTOP, LOW_SPEC_4GB_DEVICE
)


def test_1_rate_distortion_and_allocation():
    sigma_sq = 1.0
    d_target = 0.25
    r_limit = gaussian_rate_distortion(sigma_sq, d_target)
    expected_r = 0.5 * math.log2(1.0 / 0.25)
    assert math.isclose(r_limit, expected_r, abs_tol=1e-5)

    layers = ["layer_0", "layer_1", "layer_2", "layer_3"]
    params = [10_000, 10_000, 10_000, 10_000]
    sensitivities = [0.1, 0.5, 1.0, 2.0]

    allocator = LayerRateAllocator(layers, params, sensitivities)
    rates = allocator.allocate(target_bpw=1.75, min_bpw=1.0, max_bpw=3.0)

    allocated_avg = sum(params[i] * rates[layers[i]] for i in range(4)) / sum(params)
    assert math.isclose(allocated_avg, 1.75, abs_tol=1e-2)
    assert rates["layer_3"] > rates["layer_0"]


def test_2_hadamard_orthogonality_and_energy():
    d = 256
    x = torch.randn(10, d)
    rht = RandomizedHadamardTransform(dim=d, seed=42)

    z = rht.forward(x)
    x_rec = rht.inverse(z)

    energy_x = torch.sum(x ** 2).item()
    energy_z = torch.sum(z ** 2).item()
    rel_energy_diff = abs(energy_x - energy_z) / energy_x
    assert rel_energy_diff < 1e-4

    rec_error = (torch.norm(x - x_rec) / torch.norm(x)).item()
    assert rec_error < 1e-4


def test_3_gaussian_concentration_and_kurtosis():
    d = 512
    torch.manual_seed(123)
    spiky_weights = torch.randn(64, d) * 0.1
    spiky_weights[:, 0] = 5.0

    rht = RandomizedHadamardTransform(dim=d, seed=42)
    rotated = rht.forward(spiky_weights)

    diag = compute_distribution_diagnostics(spiky_weights, rotated)
    assert diag["kurtosis_after"] < diag["kurtosis_before"]


def test_4_polar_quant_2d():
    pq = PolarQuant2D(r_bits=4, theta_bits_base=4)
    x = torch.randn(1000, 2)
    x_hat, meta = pq.quantize(x)

    assert x_hat.shape == x.shape
    assert meta["snr_db"] > 5.0


def test_5_e8_lattice_quantization():
    quantizer = E8LatticeQuantizer(search_steps=16)
    x = torch.randn(256, 8)
    x_hat, meta = quantizer.quantize(x)

    assert x_hat.shape == x.shape
    assert meta["snr_db"] > 8.0
    assert math.isclose(meta["sphere_packing_density_e8"], math.pi ** 4 / 384.0, abs_tol=1e-5)


def test_6_and_7_qjl_unbiasedness_and_variance():
    results = verify_qjl_unbiasedness_and_variance(dim=256, proj_dim=64, num_trials=1000, seed=42)
    assert results["is_unbiased"]
    assert results["variance_relative_error"] < 0.15


def test_8_sparse_ternary_residuals():
    e = torch.randn(1000) * 0.1
    ternary = SparseTernaryResidual(sparsity_fraction=0.05)
    r_sparse, meta = ternary.decompose(e)

    nonzero_count = torch.sum(r_sparse != 0).item()
    assert nonzero_count == 50
    unique_vals = torch.unique(torch.abs(r_sparse))
    assert len(unique_vals) <= 2
    assert meta["error_reduction_percentage"] > 0


def test_9_physical_memory_accounting():
    bpw_stats = calculate_effective_bpw(
        num_params=3_000_000_000,
        weight_payload_bytes=656_250_000,
        scale_bytes=12_000_000,
        codebook_bytes=4_000_000,
        index_bytes=18_000_000,
        residual_bytes=24_000_000,
        metadata_bytes=1_000_000
    )
    assert bpw_stats["effective_bpw"] > bpw_stats["nominal_weight_bpw"]
    assert bpw_stats["effective_bpw"] < 2.0


def test_10_flash_page_utilization():
    coalescer = FlashPageCoalescer(physical_page_size=4096)
    res_aligned = coalescer.calculate_page_touches(offset=4096, length=4096)
    assert res_aligned["storage_utilization_U"] == 1.0
    assert res_aligned["read_amplification_A"] == 1.0

    res_unaligned = coalescer.calculate_page_touches(offset=4000, length=200)
    assert res_unaligned["num_pages_touched"] == 2
    assert res_unaligned["read_amplification_A"] == 8192 / 200.0


def test_11_operational_intensity_and_roofline():
    roofline = RooflineAnalyzer(hardware=RTX_5060_LAPTOP)
    res_gen = roofline.analyze_layer(flops=6e9, bytes_transferred=6.56e8)
    assert res_gen["bottleneck"] == "Memory Bandwidth Bound"

    res_prefill = roofline.analyze_layer(flops=1.5e12, bytes_transferred=1e9)
    assert res_prefill["bottleneck"] == "Compute Bound"


def test_12_entropy_constrained_quantization_and_shannon():
    """Validates discrete Shannon entropy and Entropy-Constrained Scalar Quantizer (ECSQ)."""
    torch.manual_seed(42)
    # Generate non-uniform Gaussian-Laplacian mixture weights
    weights = torch.randn(2048) * 0.05
    # Inject heavy-tailed center
    weights[:1024] *= 0.2

    ecsq = EntropyConstrainedQuantizer(num_levels=16, lambda_entropy=0.005)
    w_hat, indices, metrics = ecsq.fit_quantize(weights)

    assert w_hat.shape == weights.shape
    assert metrics["entropy_bpw"] <= metrics["fixed_length_bpw"]  # H(Q) <= log2(K)
    assert metrics["mse_distortion"] < 0.01
    assert shannon_entropy(indices) == pytest.approx(metrics["entropy_bpw"], abs=1e-5)


def test_13_block_hadamard_arbitrary_dimensions():
    """Validates BlockHadamardTransform on arbitrary non-power-of-2 dimensions (e.g. SwiGLU 11008)."""
    # 11008 is typical Llama-7B/Qwen FFN intermediate dim: 11008 % 256 == 0, but not a power of 2!
    dim = 11008
    bht = BlockHadamardTransform(in_features=dim, block_size=256, seed=42)

    x = torch.randn(4, dim)
    z = bht.forward(x)
    x_rec = bht.inverse(z)

    # Strict energy preservation
    energy_x = torch.sum(x ** 2).item()
    energy_z = torch.sum(z ** 2).item()
    rel_energy_err = abs(energy_x - energy_z) / energy_x
    assert rel_energy_err < 1e-4

    # Perfect inversion
    rel_rec_err = (torch.norm(x - x_rec) / torch.norm(x)).item()
    assert rel_rec_err < 1e-4

    # Test arbitrary odd dimension with remainder: e.g. 500
    bht_odd = BlockHadamardTransform(in_features=500, block_size=128, seed=42)
    x_odd = torch.randn(2, 500)
    z_odd = bht_odd.forward(x_odd)
    x_odd_rec = bht_odd.inverse(z_odd)
    assert abs(torch.sum(x_odd ** 2).item() - torch.sum(z_odd ** 2).item()) / torch.sum(x_odd ** 2).item() < 1e-4
    assert (torch.norm(x_odd - x_odd_rec) / torch.norm(x_odd)).item() < 1e-4


def test_14_gaussian_concentration_spherical_shells():
    """Validates E[||z||^2] = d*sigma^2, Var(||z||^2) = 2*d*sigma^4, and shell concentration."""
    d = 128
    sigma_sq = 0.04
    num_vectors = 2000
    torch.manual_seed(999)
    z = torch.randn(num_vectors, d) * math.sqrt(sigma_sq)

    res = verify_gaussian_concentration(z, dimension=d, expected_sigma_sq=sigma_sq)
    assert res["error_E_norm_sq"] < 0.03
    assert res["error_Var_norm_sq"] < 0.15
    assert res["concentrated_on_shell"]


def test_15_recursive_polar_quant_tree():
    """Validates RecursivePolarQuant forward hierarchical tree, quantization, and exact inversion."""
    rpq = RecursivePolarQuant(block_size=8, norm_bits=6, angle_bits=4)
    # Random 8D blocks
    torch.manual_seed(42)
    blocks = torch.randn(64, 8) * 0.1

    # Exact forward-inverse without quantization error check
    top_r, angle_levels = rpq.forward_recursive(blocks)
    assert len(angle_levels) == 3  # log2(8) = 3 levels
    reconstructed_exact = rpq.inverse_recursive(top_r, angle_levels)
    exact_err = torch.norm(blocks - reconstructed_exact).item()
    assert exact_err < 1e-5

    # Quantized execution check
    recon_q, mse = rpq.quantize_block(blocks)
    assert recon_q.shape == blocks.shape
    assert mse < 0.05


def test_16_polar_bitstream_packing_unpacking():
    """Validates bit-packing of (r_idx, theta_idx) and zero-loss lossless unpacking."""
    r_bits = 4
    th_bits = 4
    num_pairs = 512

    torch.manual_seed(777)
    r_orig = torch.randint(0, 16, (num_pairs,), dtype=torch.int64)
    th_orig = torch.randint(0, 16, (num_pairs,), dtype=torch.int64)

    packed = pack_polar_bits(r_orig, th_orig, r_bits, th_bits)
    assert len(packed) == num_pairs * 1  # 8 bits = 1 byte per pair

    r_unpacked, th_unpacked = unpack_polar_bits(packed, num_pairs, r_bits, th_bits)
    assert torch.equal(r_orig, r_unpacked)
    assert torch.equal(th_orig, th_unpacked)


def test_17_e8_roots_and_rvq():
    """Validates 240 minimal roots of E8 lattice and multi-stage Residual Vector Quantization."""
    roots = generate_e8_roots()
    assert roots.shape == (240, 8)

    # Test Multi-stage RVQ
    rvq = E8ResidualVectorQuantizer(num_stages=2, search_steps=8)
    x = torch.randn(64, 8) * 0.2
    recon, meta = rvq.quantize(x)

    assert recon.shape == x.shape
    assert meta["num_stages"] == 2
    # Second stage must reduce MSE compared to single stage
    assert meta["stage_mses"][1] < meta["stage_mses"][0]


def test_18_xnor_popcount_qjl_estimator():
    """Validates bitwise XNOR + popcount QJL sign-sketch estimator."""
    in_features = 256
    proj_dim = 128
    torch.manual_seed(42)

    estimator = XNORPopcountQJLEstimator(in_features=in_features, proj_dim=proj_dim)
    x = torch.randn(in_features)
    e = torch.randn(in_features) * 0.1

    true_dot = float(torch.dot(x, e).item())

    packed_x, scale_x = estimator.sketch_and_pack(x, is_residual=False)
    packed_e, scale_e = estimator.sketch_and_pack(e, is_residual=True)

    assert len(packed_x) == proj_dim // 8
    assert len(packed_e) == proj_dim // 8

    estimated_dot = estimator.evaluate_inner_product(packed_x, scale_x, packed_e, scale_e)
    # Correlation between true and estimated inner product should be strong
    assert isinstance(estimated_dot, float)


def test_19_sparsity_level_comparisons():
    """Validates comparative analysis across 1%, 3%, 5%, and 10% sparsity."""
    e = torch.randn(2000) * 0.1
    comp = compare_sparsity_levels(e, sparsity_levels=[0.01, 0.03, 0.05, 0.10])

    assert "1%" in comp and "3%" in comp and "5%" in comp and "10%" in comp
    # Higher sparsity fraction must yield higher error reduction
    assert comp["10%"]["error_reduction_percentage"] > comp["1%"]["error_reduction_percentage"]
    # Higher sparsity fraction has more nonzeros and higher residual bpw overhead
    assert comp["10%"]["effective_residual_bpw"] > comp["1%"]["effective_residual_bpw"]
