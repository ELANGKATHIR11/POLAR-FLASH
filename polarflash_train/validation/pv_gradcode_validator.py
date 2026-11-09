"""
POLAR-FLASH: PV-GradCode Training Validation & Optimization Suite
Validates Objective 5:
1. Frozen discrete codes during continuous V-step updates.
2. Trainable continuous scale and normalization parameters.
3. Sensitivity-guided discrete candidate code generation.
4. Strict objective-based acceptance rule (Delta L < -epsilon).
5. Validation-loss monitoring throughout training loop.
6. Automatic rollback on loss-increasing discrete candidates.
7. Optimizer memory footprint measurement.
8. Rigorous empirical comparison against LoRA and QAT.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional

from polarflash_train.gradcodes.pv_gradcode import PVGradCodeOptimizer
from polarflash_train.lora.polar_lora import PolarLoRALinear


class QuantizedLinearWithScale(nn.Module):
    """
    Quantized linear layer with continuous scale factor s and discrete codes C:
    W_hat = s * C.
    Forward: y = x @ W_hat.T.
    """
    def __init__(self, in_features: int, out_features: int, bits: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.code_min = -(2 ** (bits - 1))
        self.code_max = 2 ** (bits - 1) - 1

        # Continuous scale parameter (trainable)
        self.scale = nn.Parameter(torch.tensor([0.05], dtype=torch.float32))

        # Discrete codes (frozen during V-step)
        initial_codes = torch.randint(self.code_min, self.code_max + 1, (out_features, in_features), dtype=torch.float32)
        self.register_buffer("codes", initial_codes)

    def dequantize(self) -> torch.Tensor:
        return self.scale * self.codes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize()
        return F.linear(x, w)


class FakeQuantizedLinearQAT(nn.Module):
    """Straight-Through Estimator (STE) Quantization-Aware Training linear layer."""
    def __init__(self, in_features: int, out_features: int, bits: int = 4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.bits = bits
        self.qmax = 2 ** (bits - 1) - 1
        self.qmin = -(2 ** (bits - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.max(torch.abs(self.weight)) / float(self.qmax)
        scale = torch.clamp(scale, min=1e-8)
        # STE fake quantize
        q_weight = torch.clamp(torch.round(self.weight / scale), self.qmin, self.qmax)
        w_ste = self.weight + (q_weight * scale - self.weight).detach()
        return F.linear(x, w_ste)


def run_pv_gradcode_training_protocol(
    layer: QuantizedLinearWithScale,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    v_steps: int = 10,
    p_steps: int = 3,
    lr: float = 1e-3,
    min_improvement: float = 1e-5
) -> Dict[str, Any]:
    """
    Executes alternating PV-GradCode optimization:
    - Verifies codes are strictly frozen during V-step.
    - Demonstrates continuous convergence.
    - Demonstrates candidate generation & strict descent acceptance / rollback.
    - Monitors validation loss.
    """
    optimizer = PVGradCodeOptimizer(
        continuous_params=[layer.scale],
        discrete_codes=layer.codes,
        lr=lr,
        min_improvement=min_improvement,
        code_min=layer.code_min,
        code_max=layer.code_max
    )

    initial_codes_snapshot = layer.codes.clone()
    val_loss_history = []
    p_step_results = []

    # Compute initial validation loss
    with torch.no_grad():
        val_pred = layer(val_x)
        initial_val_loss = float(F.mse_loss(val_pred, val_y).item())
    val_loss_history.append(initial_val_loss)

    for p_cycle in range(p_steps):
        # 1. Continuous V-Steps: optimize scale with codes frozen
        for v_iter in range(v_steps):
            pred = layer(train_x)
            train_loss = F.mse_loss(pred, train_y)
            optimizer.v_step(train_loss)

        # Verification: codes must remain strictly identical to start of cycle
        assert torch.equal(layer.codes, initial_codes_snapshot), "Discrete codes mutated during continuous V-step!"

        # 2. Discrete P-Step: compute code gradients & propose candidate
        with torch.no_grad():
            train_pred = layer(train_x)
            # Analytical gradient of loss w.r.t codes: dL/dC = 2 * (x@W^T - y)^T @ x * scale
            error = train_pred - train_y
            code_grad = (2.0 / train_x.shape[0]) * (error.t() @ train_x) * layer.scale.item()

        def objective_fn(c: torch.Tensor) -> float:
            w_cand = layer.scale.item() * c
            pred_cand = F.linear(train_x, w_cand)
            return float(F.mse_loss(pred_cand, train_y).item())

        p_res = optimizer.p_step(
            objective_eval_fn=objective_fn,
            code_gradient=code_grad,
            topk_fraction=0.05
        )
        p_step_results.append(p_res)

        # Update codes snapshot if accepted
        initial_codes_snapshot = layer.codes.clone()

        # 3. Monitor validation loss
        with torch.no_grad():
            val_loss = float(F.mse_loss(layer(val_x), val_y).item())
        val_loss_history.append(val_loss)

    # 4. Verify Adversarial Rollback: propose intentionally bad candidates
    def adversarial_objective(c: torch.Tensor) -> float:
        # Intentionally returns huge loss to trigger rejection
        return 9999.0

    codes_before_bad_proposal = layer.codes.clone()
    bad_grad = torch.ones_like(layer.codes)
    bad_res = optimizer.p_step(
        objective_eval_fn=adversarial_objective,
        code_gradient=bad_grad
    )
    assert bad_res["accepted"] is False, "Adversarial loss-increasing candidate was accepted!"
    assert torch.equal(layer.codes, codes_before_bad_proposal), "Codes failed to roll back on rejected candidate!"

    memory_stats = optimizer.compute_training_memory_bytes()

    return {
        "initial_val_loss": initial_val_loss,
        "final_val_loss": val_loss_history[-1],
        "val_loss_history": val_loss_history,
        "p_step_results": p_step_results,
        "accepted_p_updates": optimizer.accepted_p_updates,
        "rejected_p_updates": optimizer.rejected_p_updates,
        "memory_breakdown": memory_stats,
        "scale_final": float(layer.scale.item())
    }


def compare_pv_gradcode_vs_lora_and_qat(
    in_features: int = 128,
    out_features: int = 128,
    num_samples: int = 500,
    steps: int = 25
) -> Dict[str, Any]:
    """
    Rigorously benchmarks PV-GradCode against LoRA and QAT:
    1. PV-GradCode: Fine-tunes quantized scale & discrete codes. 0 adapter parameters at inference.
    2. LoRA: Frozen base layer + trainable low-rank adapters (rank=8).
    3. QAT: Full float weight matrix trained with Straight-Through Estimator.
    """
    torch.manual_seed(42)
    x = torch.randn(num_samples, in_features)
    # Teacher target
    w_target = torch.randn(out_features, in_features) * 0.1
    y = F.linear(x, w_target)

    # 1. PV-GradCode
    pv_layer = QuantizedLinearWithScale(in_features, out_features, bits=4)
    pv_res = run_pv_gradcode_training_protocol(
        layer=pv_layer,
        train_x=x[:400], train_y=y[:400],
        val_x=x[400:], val_y=y[400:],
        v_steps=5, p_steps=5, lr=2e-3
    )

    # 2. LoRA
    base_linear = nn.Linear(in_features, out_features, bias=False)
    base_linear.weight.data.copy_(pv_layer.codes * pv_layer.scale.item())
    lora_layer = PolarLoRALinear(base_linear, in_features, out_features, r=8)
    lora_opt = torch.optim.AdamW([lora_layer.lora_A, lora_layer.lora_B], lr=2e-3)

    for _ in range(steps):
        lora_opt.zero_grad()
        out = lora_layer(x[:400])
        loss = F.mse_loss(out, y[:400])
        loss.backward()
        lora_opt.step()

    with torch.no_grad():
        lora_val_loss = float(F.mse_loss(lora_layer(x[400:]), y[400:]).item())

    lora_trainable_params = lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
    lora_opt_mem_bytes = lora_trainable_params * 16  # params + grads + 2 Adam moments

    # 3. QAT (Straight-Through Estimator)
    qat_layer = FakeQuantizedLinearQAT(in_features, out_features, bits=4)
    qat_opt = torch.optim.AdamW(qat_layer.parameters(), lr=2e-3)

    for _ in range(steps):
        qat_opt.zero_grad()
        out = qat_layer(x[:400])
        loss = F.mse_loss(out, y[:400])
        loss.backward()
        qat_opt.step()

    with torch.no_grad():
        qat_val_loss = float(F.mse_loss(qat_layer(x[400:]), y[400:]).item())

    qat_trainable_params = qat_layer.weight.numel()
    qat_opt_mem_bytes = qat_trainable_params * 16

    return {
        "pv_gradcode": {
            "final_val_loss": pv_res["final_val_loss"],
            "trainable_continuous_params": pv_res["memory_breakdown"]["num_continuous_parameters"],
            "optimizer_overhead_kb": pv_res["memory_breakdown"]["optimizer_bytes"] / 1024,
            "inference_adapter_overhead_bpw": 0.0  # Directly quantized in-place!
        },
        "lora": {
            "final_val_loss": lora_val_loss,
            "trainable_params": lora_trainable_params,
            "optimizer_overhead_kb": lora_opt_mem_bytes / 1024,
            "inference_adapter_overhead_bpw": (lora_trainable_params * 16.0) / (in_features * out_features)
        },
        "qat": {
            "final_val_loss": qat_val_loss,
            "trainable_params": qat_trainable_params,
            "optimizer_overhead_kb": qat_opt_mem_bytes / 1024,
            "inference_adapter_overhead_bpw": 0.0
        }
    }

