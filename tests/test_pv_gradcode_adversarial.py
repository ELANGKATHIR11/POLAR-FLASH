"""
POLAR-FLASH: Adversarial Scientific Validation Tests for PV-GradCode
Verifies:
1. Strict monotonic descent: L(c_{t+1}) < L(c_t) enforced; loss-increasing candidates rejected.
2. Alphabet clamping: discrete code transitions stay within [code_min, code_max].
3. Trust-region coordinate cap: candidate modifications do not exceed max_flips.
4. Memory safety: V-step does not retain autograd computation graph by default.
"""

import pytest
import torch
import torch.nn as nn
from polarflash_train.gradcodes.pv_gradcode import PVGradCodeOptimizer


def test_adversarial_rejection_of_increasing_loss():
    """Verifies that any candidate discrete code that increases loss is strictly rejected."""
    scale = torch.tensor([1.0], requires_grad=True)
    codes = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    optimizer = PVGradCodeOptimizer([scale], codes, lr=1e-3, min_improvement=0.0)

    # Adversarial objective: evaluates to 100.0 whenever codes are modified from original
    original_codes = codes.clone()
    def adversarial_objective(c: torch.Tensor) -> float:
        if torch.equal(c, original_codes):
            return 1.0
        return 100.0  # Huge loss degradation

    # Synthetic gradient proposing changes
    grad = torch.tensor([5.0, -5.0, 5.0, -5.0])
    res = optimizer.p_step(objective_eval_fn=adversarial_objective, code_gradient=grad)

    assert not res["accepted"], "Adversarial candidate that increases loss MUST be rejected"
    assert torch.equal(optimizer.codes, original_codes), "Codes must remain unmodified upon rejection"
    assert res["total_rejected_p_updates"] == 1
    assert res["delta_loss"] == 0.0


def test_strict_monotonic_descent_acceptance():
    """Verifies that only candidates strictly reducing loss by at least min_improvement are accepted."""
    scale = torch.tensor([1.0], requires_grad=True)
    codes = torch.tensor([5.0, 5.0], dtype=torch.float32)
    min_improvement = 0.5
    optimizer = PVGradCodeOptimizer([scale], codes, lr=1e-3, min_improvement=min_improvement)

    # Candidate reduces loss by only 0.2 (below 0.5 threshold) -> should reject
    def marginal_objective(c: torch.Tensor) -> float:
        if torch.equal(c, torch.tensor([5.0, 5.0])):
            return 10.0
        return 9.8  # Reduction is 0.2 < 0.5

    grad = torch.tensor([2.0, 2.0])
    res = optimizer.p_step(objective_eval_fn=marginal_objective, code_gradient=grad)
    assert not res["accepted"], "Candidate failing min_improvement threshold must be rejected"

    # Candidate reduces loss by 1.0 (exceeds 0.5 threshold) -> should accept
    def sufficient_objective(c: torch.Tensor) -> float:
        if torch.equal(c, torch.tensor([5.0, 5.0])):
            return 10.0
        return 9.0  # Reduction is 1.0 > 0.5

    res2 = optimizer.p_step(objective_eval_fn=sufficient_objective, code_gradient=grad)
    assert res2["accepted"], "Candidate exceeding min_improvement threshold must be accepted"
    assert res2["delta_loss"] == -1.0


def test_discrete_alphabet_bounds_clamping():
    """Verifies that candidate discrete codes are strictly clamped to [code_min, code_max]."""
    scale = torch.tensor([1.0], requires_grad=True)
    codes = torch.tensor([7.0, -8.0], dtype=torch.float32)
    optimizer = PVGradCodeOptimizer(
        [scale], codes, lr=1e-3, code_min=-8, code_max=7, min_improvement=0.0
    )

    # Propose flips that would push 7 -> 8 and -8 -> -9
    grad = torch.tensor([-10.0, 10.0])  # flip_dir: +1 for 7, -1 for -8
    evaluated_proposals = []

    def tracking_objective(c: torch.Tensor) -> float:
        # Check if evaluating proposal vs initial codes
        if not torch.equal(c, torch.tensor([7.0, -8.0])):
            evaluated_proposals.append(c.clone())
            return 0.2  # Reduced loss
        return 1.0  # Baseline loss

    res = optimizer.p_step(objective_eval_fn=tracking_objective, code_gradient=grad)
    # The proposal 7 + 1 is clamped to 7, and -8 - 1 is clamped to -8.
    # Because both coordinates were clamped to their current values,
    # the candidate code is identical to current codes, so loss does not strictly decrease.
    # Therefore, the optimizer correctly rejects a zero-improvement transition.
    assert not res["accepted"]


def test_discrete_alphabet_transition_within_bounds():
    """Verifies valid transitions within bounds are accepted and clamped properly."""
    scale = torch.tensor([1.0], requires_grad=True)
    codes = torch.tensor([6.0, -7.0], dtype=torch.float32)
    optimizer = PVGradCodeOptimizer(
        [scale], codes, lr=1e-3, code_min=-8, code_max=7, min_improvement=0.0, trust_region_radius=1.0
    )

    # Propose flips: 6 -> 7 (valid), -7 -> -8 (valid)
    grad = torch.tensor([-10.0, 10.0])

    def objective(c: torch.Tensor) -> float:
        if torch.equal(c, torch.tensor([6.0, -7.0])):
            return 1.0
        return 0.2

    res = optimizer.p_step(objective_eval_fn=objective, code_gradient=grad, topk_fraction=1.0)
    assert res["accepted"]
    assert optimizer.codes[0].item() == 7.0
    assert optimizer.codes[1].item() == -8.0


def test_autograd_graph_retention_safety():
    """Verifies that v_step does not retain computation graph across steps when retain_graph=False."""
    scale = torch.tensor([2.0], requires_grad=True)
    codes = torch.tensor([1.0, 2.0, 3.0])
    optimizer = PVGradCodeOptimizer([scale], codes, lr=1e-2)

    # First step
    x = torch.tensor([3.0, 4.0, 5.0])
    loss1 = torch.sum((x * scale - codes) ** 2)
    optimizer.v_step(loss1, retain_graph=False)

    # Attempting backward on loss1 again should fail because graph was freed
    with pytest.raises(RuntimeError):
        loss1.backward()


def test_trust_region_max_flips():
    """Verifies that max_flips enforces a hard bound on the number of discrete modifications."""
    scale = torch.tensor([1.0], requires_grad=True)
    codes = torch.zeros(100, dtype=torch.float32)
    optimizer = PVGradCodeOptimizer([scale], codes, lr=1e-3, max_flips=5, trust_region_radius=0.5)

    captured = []
    def record_objective(c: torch.Tensor) -> float:
        captured.append(c.clone())
        if len(captured) == 1:
            return 1.0  # baseline
        return 0.1  # improved

    grad = torch.ones(100) * 2.0
    res = optimizer.p_step(objective_eval_fn=record_objective, code_gradient=grad)
    assert res["accepted"]
    assert res["k_coordinates_flipped"] == 5
    # captured[1] is the candidate proposal
    diff = torch.sum(captured[1] != torch.zeros(100)).item()
    assert diff == 5, f"Expected exactly 5 coordinates flipped, got {diff}"
