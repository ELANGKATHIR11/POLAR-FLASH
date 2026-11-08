"""
POLAR-FLASH: Information-Theoretic Rate-Distortion & Bit Allocation Engine
Implements Part II Section 1 of the POLAR-FLASH Mathematical Specification:
1. Empirical entropy estimation of weight tensors.
2. Gaussian & vector-source rate-distortion limits: R(D) = 0.5 * sum(max(0, log2(lambda_i / theta))).
3. Constrained Lagrangian bit allocation across transformer layers.
4. Serialized bits-per-weight accounting including metadata, codebooks, scales, and indices.
"""

import math
import torch
import numpy as np
from typing import Dict, List, Tuple, Any


def empirical_entropy(tensor: torch.Tensor, bins: int = 256) -> float:
    """
    Computes the empirical Shannon entropy H(X) = -sum(p * log2(p)) in bits.
    """
    flat = tensor.detach().cpu().float().view(-1)
    if flat.numel() == 0:
        return 0.0

    counts = torch.histc(flat, bins=bins, min=flat.min().item(), max=flat.max().item())
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -torch.sum(probs * torch.log2(probs)).item()


def gaussian_rate_distortion(sigma_sq: float, distortion: float) -> float:
    """
    Scalar Gaussian rate-distortion lower bound:
    R(D) = 0.5 * log2(sigma^2 / D) for 0 < D <= sigma^2.
    """
    if distortion <= 0:
        return float("inf")
    if distortion >= sigma_sq:
        return 0.0
    return 0.5 * math.log2(sigma_sq / distortion)


def vector_source_rate_distortion(eigenvalues: np.ndarray, target_distortion: float) -> Tuple[float, float]:
    """
    Vector Gaussian Rate-Distortion via reverse water-filling:
    D(theta) = sum_i min(theta, lambda_i)
    R(theta) = 0.5 * sum_i max(0, log2(lambda_i / theta))
    """
    lambdas = np.sort(eigenvalues)[::-1]
    n = len(lambdas)

    # Binary search for water level theta such that sum(min(theta, lambda_i)) == target_distortion
    low, high = 1e-12, float(np.max(lambdas))
    theta = low

    for _ in range(64):
        mid = (low + high) / 2.0
        current_d = np.sum(np.minimum(mid, lambdas))
        if current_d < target_distortion:
            low = mid
        else:
            high = mid
        theta = mid

    rate = 0.5 * np.sum(np.maximum(0.0, np.log2(lambdas / (theta + 1e-15)))) / n
    distortion = float(np.sum(np.minimum(theta, lambdas))) / n
    return rate, distortion


class LayerRateAllocator:
    """
    Solves the constrained layer-rate allocation problem:
    min sum_l alpha_l * D_l(R_l) subject to sum_l N_l * R_l <= B_total
    using Lagrangian relaxation: L(R, lambda) = sum_l alpha_l * D_l(R_l) + lambda * (sum_l N_l * R_l - B)
    """
    def __init__(self, layer_names: List[str], param_counts: List[int], sensitivities: List[float]):
        self.layer_names = layer_names
        self.param_counts = np.array(param_counts, dtype=np.float64)
        self.sensitivities = np.array(sensitivities, dtype=np.float64)
        self.total_params = np.sum(self.param_counts)

    def allocate(self, target_bpw: float = 1.75, min_bpw: float = 1.0, max_bpw: float = 3.5) -> Dict[str, float]:
        """
        Allocates fractional bitrates {R_l} per layer proportional to sensitivity alpha_l
        while strictly adhering to the total parameter bit budget B_total = target_bpw * total_params.
        """
        total_budget = target_bpw * self.total_params

        # Closed-form Lagrangian allocation under exponential rate-distortion model D(R) ~ sigma^2 * 2^(-2R):
        # R_l = R_avg + 0.5 * log2(alpha_l / geometric_mean(alpha))
        log_alpha = np.log2(np.maximum(self.sensitivities, 1e-6))
        weighted_log_alpha = np.sum(self.param_counts * log_alpha) / self.total_params

        allocated_rates = target_bpw + 0.5 * (log_alpha - weighted_log_alpha)
        # Clamp to physical hardware feasibility bounds
        allocated_rates = np.clip(allocated_rates, min_bpw, max_bpw)

        # Scale correction to satisfy exact budget constraint
        actual_bits = np.sum(self.param_counts * allocated_rates)
        scale = total_budget / actual_bits
        final_rates = allocated_rates * scale

        return {name: float(rate) for name, rate in zip(self.layer_names, final_rates)}


def calculate_effective_bpw(
    num_params: int,
    weight_payload_bytes: int,
    scale_bytes: int,
    codebook_bytes: int,
    index_bytes: int,
    residual_bytes: int,
    metadata_bytes: int
) -> Dict[str, float]:
    """
    Computes true physical effective bits-per-weight (bpw):
    b_effective = 8 * S_total / N
    Strictly accounts for all overhead: scales, codebooks, indices, residuals, and metadata.
    """
    total_bytes = (
        weight_payload_bytes +
        scale_bytes +
        codebook_bytes +
        index_bytes +
        residual_bytes +
        metadata_bytes
    )
    effective_bpw = (8.0 * total_bytes) / max(num_params, 1)

    return {
        "num_params": num_params,
        "total_bytes": total_bytes,
        "effective_bpw": effective_bpw,
        "nominal_weight_bpw": (8.0 * weight_payload_bytes) / max(num_params, 1),
        "overhead_bpw": (8.0 * (total_bytes - weight_payload_bytes)) / max(num_params, 1),
        "overhead_percentage": (1.0 - (weight_payload_bytes / max(total_bytes, 1))) * 100.0
    }
