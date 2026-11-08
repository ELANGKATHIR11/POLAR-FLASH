"""
POLAR-FLASH: Scientific Correctness Verification for E8 Lattice Quantizer
Validates:
1. Complete Gosset lattice E8 = D8 U (D8 + 0.5 * 1_8) decoder for arbitrary R^8 points.
2. Verification against covering radius (R <= 1.0 for all points in R^8).
3. Parity and integral/half-integral coset constraints:
   - Either all coordinates in Z and sum(x_i) = 0 mod 2
   - Or all coordinates in Z + 0.5 and sum(x_i) = 0 mod 2
4. Boundary vectors, negative coordinates, large magnitude vectors (1e5).
5. Coset boundary vectors and tied nearest points.
6. Distinction between 240 minimal root vectors (Shell 1: ||v||^2 = 2.0) and full infinite lattice.
"""

import math
import pytest
import torch
from polarflash_quant.lattice.e8_lattice import (
    quantize_d8, quantize_e8, E8LatticeQuantizer, generate_e8_roots
)


def assert_valid_e8_point(y: torch.Tensor, atol: float = 1e-5):
    """
    Validates that point y lies in the infinite Gosset lattice E8:
    1. 2 * y in Z^8 (all coordinates integers or all half-integers).
    2. Fractional parts identical across all 8 dimensions.
    3. sum(y_i) is an even integer.
    """
    assert y.shape[-1] == 8
    flat_y = y.view(-1, 8)
    for vec in flat_y:
        # Check 2*vec is integer
        two_vec = vec * 2.0
        diff_from_int = torch.abs(two_vec - torch.round(two_vec))
        assert torch.all(diff_from_int < atol), f"Vector {vec} coordinates are not half-integers or integers"

        # Check all coords have same fractional part (either ~0.0 or ~0.5)
        frac = torch.abs(vec - torch.round(vec))
        is_all_int = torch.all(frac < atol)
        is_all_half = torch.all(torch.abs(frac - 0.5) < atol)
        assert is_all_int or is_all_half, f"Vector {vec} does not belong to standard E8 cosets"

        # Check sum is an even integer
        sum_val = torch.sum(vec).item()
        nearest_even = round(sum_val / 2.0) * 2
        assert abs(sum_val - nearest_even) < atol, f"Sum of vector coordinates {sum_val} is not an even integer"


def independent_reference_d8(x: torch.Tensor) -> torch.Tensor:
    """Independent reference D8 decoder using brute-force error ranking."""
    f = torch.round(x)
    s = int(torch.sum(f).item())
    if s % 2 == 0:
        return f
    # Must flip one coordinate to second nearest integer
    errors = []
    for i in range(8):
        xi = x[i].item()
        fi = f[i].item()
        alt = fi + 1.0 if xi >= fi else fi - 1.0
        err_increase = (xi - alt) ** 2 - (xi - fi) ** 2
        errors.append((err_increase, i, alt))
    errors.sort(key=lambda t: t[0])
    best_alt_idx = errors[0][1]
    best_alt_val = errors[0][2]
    out = f.clone()
    out[best_alt_idx] = best_alt_val
    return out


def independent_reference_e8(x: torch.Tensor) -> torch.Tensor:
    """Independent reference E8 decoder."""
    cand1 = independent_reference_d8(x)
    cand2 = independent_reference_d8(x - 0.5) + 0.5
    d1 = torch.sum((x - cand1) ** 2).item()
    d2 = torch.sum((x - cand2) ** 2).item()
    return cand1 if d1 <= d2 else cand2


def test_random_vectors_against_reference():
    """Validates quantize_e8 against independent reference decoder across 500 random vectors."""
    torch.manual_seed(42)
    x = torch.randn(500, 8)
    q_fast = quantize_e8(x)

    for i in range(500):
        vec = x[i]
        q_ref = independent_reference_e8(vec)
        q_act = q_fast[i]
        d_act = torch.sum((vec - q_act) ** 2).item()
        d_ref = torch.sum((vec - q_ref) ** 2).item()
        # In case of tied nearest points, distances must be identical
        assert math.isclose(d_act, d_ref, abs_tol=1e-5), f"Discrepancy: d_act={d_act}, d_ref={d_ref}"
        assert_valid_e8_point(q_act)


def test_e8_covering_radius_bound():
    """
    Theorem: The covering radius of Gosset lattice E8 is exactly R = 1.0.
    For any vector x in R^8: min_{y in E8} ||x - y||_2 <= 1.0.
    """
    torch.manual_seed(123)
    x = torch.randn(1000, 8) * 10.0
    q = quantize_e8(x)
    distances = torch.sqrt(torch.sum((x - q) ** 2, dim=-1))
    max_dist = float(torch.max(distances).item())
    assert max_dist <= 1.0 + 1e-5, f"Covering radius violated: max distance = {max_dist} > 1.0"


def test_large_magnitude_vectors():
    """Validates E8 quantizer on large-magnitude vectors (1e5 to 1e7) without overflow."""
    x = (torch.randn(100, 8) + 1.0) * 1e5
    q = quantize_e8(x)
    assert_valid_e8_point(q)
    distances = torch.sqrt(torch.sum((x - q) ** 2, dim=-1))
    assert torch.all(distances <= 1.0 + 1e-4)


def test_negative_coordinates():
    """Validates E8 quantizer on strictly negative coordinate vectors."""
    x = -torch.rand(100, 8) * 50.0 - 1.0
    q = quantize_e8(x)
    assert_valid_e8_point(q)
    distances = torch.sqrt(torch.sum((x - q) ** 2, dim=-1))
    assert torch.all(distances <= 1.0 + 1e-5)


def test_boundary_and_coset_boundary_vectors():
    """Validates points situated exactly on half-integer boundaries and coset boundaries."""
    # Point equidistant from D8 and (D8 + 0.5)
    # e.g., x = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
    x_boundary = torch.full((1, 8), 0.25)
    q = quantize_e8(x_boundary)
    assert_valid_e8_point(q)
    d = torch.sum((x_boundary - q) ** 2).item()
    # Distance to 0 is 8 * 0.25^2 = 0.5. Distance to 0.5*1 is 8 * 0.25^2 = 0.5. Tied!
    assert math.isclose(d, 0.5, abs_tol=1e-5)

    # Half integer coordinates: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] is an exact E8 lattice point!
    x_half = torch.full((1, 8), 0.5)
    q_half = quantize_e8(x_half)
    assert torch.allclose(q_half, x_half)
    assert_valid_e8_point(q_half)


def test_root_vectors_distinct_from_infinite_lattice():
    """
    Verifies that generate_e8_roots returns the 240 minimal vectors of Shell 1 (norm^2 = 2.0),
    which is a finite subset, whereas quantize_e8 quantizes onto the infinite lattice.
    """
    roots = generate_e8_roots()
    assert roots.shape == (240, 8)
    # Every root must be a valid E8 lattice point
    assert_valid_e8_point(roots)
    norms_sq = torch.sum(roots ** 2, dim=-1)
    assert torch.allclose(norms_sq, torch.full_like(norms_sq, 2.0))

    # A point with norm > 2 like [2, 0, 0, 0, 0, 0, 0, 0] is in E8 but NOT in the 240 minimal roots
    large_pt = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    q_large = quantize_e8(large_pt)
    assert torch.allclose(q_large, large_pt)
    assert_valid_e8_point(q_large)
    # Verify large_pt is not in the 240 roots
    matches = torch.all(torch.isclose(roots, large_pt), dim=-1)
    assert not torch.any(matches)

