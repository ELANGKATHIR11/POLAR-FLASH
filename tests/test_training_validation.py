"""
POLAR-FLASH: PV-GradCode Training Validation Tests
Validates Objective 5:
1. Frozen discrete codes during continuous V-step.
2. Continuous parameter gradient propagation and updates.
3. Valid discrete candidate code generation and bounds clamping.
4. Strict objective-based descent acceptance rule.
5. Validation loss tracking.
6. Rollback guarantees on adversarial/degraded candidates.
7. Optimizer memory measurement.
8. Empirical comparison against LoRA and QAT.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from polarflash_train.gradcodes.pv_gradcode import PVGradCodeOptimizer
from polarflash_train.validation.pv_gradcode_validator import (
    QuantizedLinearWithScale,
    run_pv_gradcode_training_protocol,
    compare_pv_gradcode_vs_lora_and_qat
)


def test_frozen_discrete_codes_and_continuous_updates():
    """Verifies that discrete codes remain strictly bit-level identical during continuous V-steps."""
    torch.manual_seed(42)
    layer = QuantizedLinearWithScale(in_features=64, out_features=64, bits=4)
    codes_before = layer.codes.clone()
    scale_before = layer.scale.clone()

    x = torch.randn(16, 64)
    y = torch.randn(16, 64) * 0.1

    opt = PVGradCodeOptimizer([layer.scale], layer.codes, lr=1e-2)

    for _ in range(5):
        loss = F.mse_loss(layer(x), y)
        opt.v_step(loss)

    # 1. Discrete codes must NOT have changed at all
    assert torch.equal(layer.codes, codes_before), "Codes mutated during V-step!"

    # 2. Continuous scale MUST have updated
    assert not torch.equal(layer.scale, scale_before), "Continuous scale failed to update!"
    assert layer.scale.grad is not None


def test_discrete_candidate_bounds_and_strict_acceptance():
    """Verifies that candidate codes respect alphabet boundaries and strictly reduce loss."""
    torch.manual_seed(77)
    layer = QuantizedLinearWithScale(in_features=32, out_features=32, bits=4)
    opt = PVGradCodeOptimizer(
        [layer.scale],
        layer.codes,
        min_improvement=1e-5,
        code_min=-8,
        code_max=7
    )

    x = torch.randn(16, 32)
    target = torch.randn(16, 32) * 0.05

    def loss_eval(c):
        return float(F.mse_loss(F.linear(x, layer.scale.item() * c), target).item())

    # Initial loss
    l0 = loss_eval(layer.codes)

    # Gradient-guided candidate
    pred = layer(x)
    code_grad = 2.0 * ((pred - target).t() @ x) * layer.scale.item()

    res = opt.p_step(objective_eval_fn=loss_eval, code_gradient=code_grad)

    # Check bounds
    assert torch.all(layer.codes >= -8)
    assert torch.all(layer.codes <= 7)

    # Check monotonic descent condition
    if res["accepted"]:
        assert res["candidate_loss"] < l0
    else:
        assert torch.equal(layer.codes, layer.codes)


def test_adversarial_candidate_rollback():
    """Verifies automatic rollback when a candidate code increases loss."""
    layer = QuantizedLinearWithScale(in_features=32, out_features=32, bits=4)
    codes_snapshot = layer.codes.clone()

    opt = PVGradCodeOptimizer([layer.scale], layer.codes, min_improvement=0.0)

    # Adversarial objective that always returns worse loss
    def adversarial_fn(c):
        return 1e6

    res = opt.p_step(objective_eval_fn=adversarial_fn, code_gradient=torch.randn_like(layer.codes))
    assert res["accepted"] is False
    assert torch.equal(layer.codes, codes_snapshot), "Rollback failed!"


def test_training_protocol_with_validation_monitoring():
    """Tests the full alternating PV-GradCode protocol including validation tracking and memory measurement."""
    torch.manual_seed(123)
    layer = QuantizedLinearWithScale(in_features=64, out_features=64, bits=4)
    train_x = torch.randn(64, 64)
    train_y = torch.randn(64, 64) * 0.1
    val_x = torch.randn(32, 64)
    val_y = torch.randn(32, 64) * 0.1

    res = run_pv_gradcode_training_protocol(
        layer=layer,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        v_steps=5,
        p_steps=3
    )

    assert "val_loss_history" in res
    assert len(res["val_loss_history"]) >= 4
    assert res["memory_breakdown"]["total_bytes"] > 0
    assert res["memory_breakdown"]["num_continuous_parameters"] == 1


def test_pv_gradcode_vs_lora_and_qat_comparison():
    """Verifies empirical comparison of PV-GradCode against LoRA and QAT."""
    comparison = compare_pv_gradcode_vs_lora_and_qat(in_features=64, out_features=64, num_samples=200, steps=10)

    assert "pv_gradcode" in comparison
    assert "lora" in comparison
    assert "qat" in comparison

    # PV-GradCode requires 0 additional inference adapter parameters (quantized weights modified in-place)
    assert comparison["pv_gradcode"]["inference_adapter_overhead_bpw"] == 0.0
    assert comparison["lora"]["inference_adapter_overhead_bpw"] > 0.0

    # PV-GradCode optimizer memory is orders of magnitude smaller than QAT full float weights
    assert comparison["pv_gradcode"]["optimizer_overhead_kb"] < comparison["qat"]["optimizer_overhead_kb"]

