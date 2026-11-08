"""
POLAR-FLASH: Statistical Monte Carlo Validation for Quantized Johnson-Lindenstrauss (QJL)
Validates:
1. Floating-point Rademacher estimator:
   E[<x, e>_hat] = <x, e> (unbiasedness)
   Var(<x, e>_hat) = (1/m) * (||x||^2 ||e||^2 + <x,e>^2 - 2 * sum x_i^2 e_i^2).
2. Bit-Packed Rademacher estimator: exact equivalence with floating-point Rademacher.
3. 1-Bit Sign-Sketch XNOR + Popcount estimator with Grothendieck arc-sine correction.
4. Explicit distinction between linear Rademacher and nonlinear sign-sketch estimators.
"""

import math
import pytest
import torch
from polarflash_quant.residual.qjl import (
    QJLResidualEstimator, BitPackedQJLEstimator, XNORPopcountQJLEstimator,
    verify_qjl_unbiasedness_and_variance
)


def test_rademacher_unbiasedness_and_variance_monte_carlo():
    """
    Validates unbiasedness and exact theoretical variance formula of Rademacher QJL
    using 1,000 independent Monte Carlo projection trials.
    """
    res = verify_qjl_unbiasedness_and_variance(
        dim=256, proj_dim=64, num_trials=1000, seed=42
    )
    assert res["is_unbiased"], f"Estimator is biased! Z-score: {res['z_score']:.3f}"
    assert res["variance_relative_error"] < 0.15, (
        f"Empirical variance deviates from theory by {res['variance_relative_error']*100:.1f}%"
    )


def test_bit_packed_qjl_equivalence():
    """Verifies BitPackedQJLEstimator produces bit-identical projections to QJLResidualEstimator."""
    dim = 128
    proj_dim = 64
    seed = 999

    ref_est = QJLResidualEstimator(in_features=dim, proj_dim=proj_dim, seed=seed)
    packed_est = BitPackedQJLEstimator(in_features=dim, proj_dim=proj_dim, seed=seed)

    x = torch.randn(4, dim)
    e = torch.randn(4, dim) * 0.1

    ref_sx = ref_est.project_activation(x)
    packed_sx = packed_est.project_activation(x)
    assert torch.allclose(ref_sx, packed_sx, atol=1e-6)

    ref_se = ref_est.project_residual(e)
    packed_se = packed_est.project_residual(e)
    assert torch.allclose(ref_se, packed_se, atol=1e-6)

    ref_dot = ref_est.estimate_inner_product(ref_sx, ref_se)
    packed_dot = packed_est.estimate_inner_product(packed_sx, packed_se)
    assert torch.allclose(ref_dot, packed_dot, atol=1e-6)


def test_sign_sketch_xnor_popcount_monte_carlo():
    """
    Validates that the 1-bit sign-sketch with Grothendieck sine-correction
    converges to the true inner product over Monte Carlo trials.
    """
    dim = 256
    proj_dim = 128
    torch.manual_seed(777)

    x = torch.randn(dim)
    e = torch.randn(dim) * 0.2
    true_inner = float(torch.dot(x, e).item())

    estimates = []
    num_trials = 600
    for trial in range(num_trials):
        estimator = XNORPopcountQJLEstimator(
            in_features=dim, proj_dim=proj_dim, seed=1000 + trial
        )
        packed_x, scale_x = estimator.sketch_and_pack(x, is_residual=False)
        packed_e, scale_e = estimator.sketch_and_pack(e, is_residual=True)
        est = estimator.evaluate_inner_product(packed_x, scale_x, packed_e, scale_e)
        estimates.append(est)

    mean_est = sum(estimates) / len(estimates)
    bias = abs(mean_est - true_inner)
    rel_bias = bias / (abs(true_inner) + 1e-6)

    # For sign sketch, mean estimate over 600 trials should be within 15% relative of true inner product
    assert rel_bias < 0.15, f"Sign-sketch bias too large: mean={mean_est:.4f}, true={true_inner:.4f}, rel={rel_bias:.3f}"

