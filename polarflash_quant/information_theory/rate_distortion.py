"""
POLAR-FLASH: Information-Theoretic Rate-Distortion & Bit Allocation Engine
Implements Part II Section 1 of the POLAR-FLASH Mathematical Specification:
1. Discrete Shannon entropy H(Q) and continuous histogram empirical entropy.
2. Scalar Gaussian R(D) = 0.5 * log2(sigma^2 / D) & Vector Gaussian reverse water-filling.
3. Entropy-Constrained Scalar Quantization (ECSQ) via generalized Lloyd-Max with Lagrangian J = D + lambda * H.
4. Empirical non-Gaussian rate-distortion curve estimation on real model weights.
5. Constrained Lagrangian bit allocation across transformer layers (supporting both empirical RD curves and sensitivity weights).
6. Exact physical effective bits-per-weight accounting (scales, codebooks, indices, residuals, metadata).
"""

import math
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def shannon_entropy(indices: torch.Tensor) -> float:
    """
    Computes exact discrete Shannon entropy H(Q) = -sum_k p_k log2(p_k) in bits per symbol
    for a tensor of quantized integer indices or discrete symbols.
    """
    flat = indices.detach().cpu().view(-1)
    n = flat.numel()
    if n == 0:
        return 0.0
    _, counts = torch.unique(flat, return_counts=True)
    probs = counts.float() / float(n)
    probs = probs[probs > 0]
    return float(-torch.sum(probs * torch.log2(probs)).item())


def empirical_entropy(tensor: torch.Tensor, bins: int = 256) -> float:
    """
    Computes the empirical Shannon entropy H(X) = -sum(p * log2(p)) in bits
    using a histogram estimator over continuous tensor values.
    """
    flat = tensor.detach().cpu().float().view(-1)
    if flat.numel() == 0:
        return 0.0
    vmin, vmax = flat.min().item(), flat.max().item()
    if abs(vmax - vmin) < 1e-12:
        return 0.0

    counts = torch.histc(flat, bins=bins, min=vmin, max=vmax)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-torch.sum(probs * torch.log2(probs)).item())


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
    Returns (rate_per_dim, achieved_distortion_per_dim).
    """
    lambdas = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
    n = len(lambdas)
    if n == 0:
        return 0.0, 0.0

    total_var = float(np.sum(lambdas))
    if target_distortion >= total_var:
        return 0.0, total_var / n
    if target_distortion <= 0:
        return float("inf"), 0.0

    # Binary search for water level theta such that sum(min(theta, lambda_i)) == target_distortion
    low, high = 1e-15, float(np.max(lambdas))
    theta = low

    for _ in range(80):
        mid = (low + high) / 2.0
        current_d = float(np.sum(np.minimum(mid, lambdas)))
        if current_d < target_distortion:
            low = mid
        else:
            high = mid
        theta = mid

    rate = float(0.5 * np.sum(np.maximum(0.0, np.log2(lambdas / max(theta, 1e-15)))) / n)
    distortion = float(np.sum(np.minimum(theta, lambdas)) / n)
    return rate, distortion


class EntropyConstrainedQuantizer:
    """
    Entropy-Constrained Scalar Quantizer (ECSQ).
    Minimizes J(C, p) = E[(w - c_{q(w)})^2] + lambda_ent * E[-log2 p(q(w))]
    via alternating nearest-entropy-penalized assignment and centroid/probability updates.
    Does not assume Gaussianity; adapts directly to the empirical weight distribution.
    """
    def __init__(self, num_levels: int = 16, lambda_entropy: float = 0.001, max_iters: int = 25):
        self.num_levels = num_levels
        self.lambda_entropy = lambda_entropy
        self.max_iters = max_iters
        self.centroids: Optional[torch.Tensor] = None
        self.probs: Optional[torch.Tensor] = None

    def fit_quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Fits ECSQ codebook to tensor x and returns (reconstructed_x, indices, metrics).
        """
        orig_shape = x.shape
        flat = x.detach().float().view(-1)
        n = flat.numel()

        # Initialize centroids using empirical quantiles to handle heavy-tailed / super-Gaussian weights
        quantiles = torch.linspace(0.01, 0.99, self.num_levels, device=flat.device)
        centroids = torch.quantile(flat, quantiles)
        probs = torch.full((self.num_levels,), 1.0 / self.num_levels, device=flat.device)

        # Subsample for fast iterative fitting if tensor is very large
        if n > 65536:
            gen = torch.Generator(device=flat.device).manual_seed(42)
            perm = torch.randperm(n, generator=gen, device=flat.device)[:65536]
            sample = flat[perm]
        else:
            sample = flat

        for _ in range(self.max_iters):
            # Rate penalty: -lambda * log2(p_k)
            rate_cost = -self.lambda_entropy * torch.log2(torch.clamp(probs, min=1e-9))
            # Distortion + Rate cost matrix: (N_sample, K)
            dist_sq = (sample.unsqueeze(1) - centroids.unsqueeze(0)) ** 2
            total_cost = dist_sq + rate_cost.unsqueeze(0)
            assign = torch.argmin(total_cost, dim=1)

            # Update centroids and symbol probabilities
            new_counts = torch.bincount(assign, minlength=self.num_levels).float()
            active = new_counts > 0
            if torch.any(active):
                sums = torch.zeros(self.num_levels, device=flat.device)
                sums.scatter_add_(0, assign, sample)
                centroids = torch.where(active, sums / torch.clamp(new_counts, min=1.0), centroids)
                probs = torch.clamp(new_counts / float(sample.numel()), min=1e-9)
                probs = probs / probs.sum()

        self.centroids = centroids
        self.probs = probs

        # Final assignment over full tensor in chunks to bound memory
        chunk_size = 262144
        indices = torch.empty(n, dtype=torch.int64, device=flat.device)
        rate_cost = -self.lambda_entropy * torch.log2(torch.clamp(probs, min=1e-9))
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            sub = flat[start:end]
            cost = (sub.unsqueeze(1) - centroids.unsqueeze(0)) ** 2 + rate_cost.unsqueeze(0)
            indices[start:end] = torch.argmin(cost, dim=1)

        x_hat = centroids[indices].reshape(orig_shape).to(x.dtype)
        mse = float(torch.mean((flat - centroids[indices]) ** 2).item())
        h_bits = shannon_entropy(indices)
        var_x = float(torch.var(flat).item())
        gaussian_bound_bits = gaussian_rate_distortion(max(var_x, 1e-12), max(mse, 1e-12))

        metrics = {
            "num_levels": self.num_levels,
            "active_levels": int(torch.unique(indices).numel()),
            "mse_distortion": mse,
            "entropy_bpw": h_bits,
            "fixed_length_bpw": math.log2(max(self.num_levels, 2)),
            "gaussian_rd_lower_bound_bpw": gaussian_bound_bits,
            "lagrangian_cost": mse + self.lambda_entropy * h_bits
        }
        return x_hat, indices.reshape(orig_shape), metrics


def estimate_empirical_rd_curve(
    weight_tensor: torch.Tensor,
    bit_budgets: List[int] = [1, 2, 3, 4, 6, 8]
) -> Dict[str, Any]:
    """
    Empirically measures the real Rate-Distortion curve of an arbitrary model weight tensor
    without assuming Gaussianity, comparing uniform affine quantization vs. Gaussian R(D) theory.
    """
    flat = weight_tensor.detach().cpu().float().view(-1)
    var_w = float(torch.var(flat).item())
    mean_w = float(torch.mean(flat).item())
    std_w = max(math.sqrt(var_w), 1e-12)
    kurtosis = float(torch.mean(((flat - mean_w) / std_w) ** 4).item())

    points = []
    vmin, vmax = float(flat.min().item()), float(flat.max().item())
    span = max(vmax - vmin, 1e-12)

    for b in bit_budgets:
        levels = 2 ** b
        scale = span / max(levels - 1, 1)
        q_idx = torch.clamp(torch.round((flat - vmin) / scale), 0, levels - 1).to(torch.int32)
        w_hat = q_idx.float() * scale + vmin
        mse = float(torch.mean((flat - w_hat) ** 2).item())
        h_emp = shannon_entropy(q_idx)
        r_gauss = gaussian_rate_distortion(var_w, max(mse, 1e-15))
        points.append({
            "nominal_bits": b,
            "shannon_entropy_bits": h_emp,
            "empirical_mse": mse,
            "gaussian_theoretical_rate_for_mse": r_gauss,
            "snr_db": 10.0 * math.log10(max(var_w, 1e-15) / max(mse, 1e-15))
        })

    return {
        "variance": var_w,
        "kurtosis": kurtosis,
        "is_non_gaussian": abs(kurtosis - 3.0) > 0.25,
        "rd_curve": points
    }


class LayerRateAllocator:
    """
    Solves the constrained layer-rate allocation problem:
    min sum_l alpha_l * D_l(R_l) subject to sum_l N_l * R_l <= B_total
    using Lagrangian relaxation: L(R, lambda) = sum_l alpha_l * D_l(R_l) + lambda * (sum_l N_l * R_l - B)
    Supports both analytical high-resolution asymptotic allocation and discrete candidate rate selection.
    """
    def __init__(
        self,
        layer_names: List[str],
        param_counts: List[int],
        sensitivities: List[float],
        layer_variances: Optional[List[float]] = None
    ):
        self.layer_names = layer_names
        self.param_counts = np.array(param_counts, dtype=np.float64)
        self.sensitivities = np.array(sensitivities, dtype=np.float64)
        self.variances = (
            np.array(layer_variances, dtype=np.float64)
            if layer_variances is not None
            else np.ones_like(self.sensitivities)
        )
        self.total_params = float(np.sum(self.param_counts))

    def allocate(self, target_bpw: float = 1.75, min_bpw: float = 1.0, max_bpw: float = 3.5) -> Dict[str, float]:
        """
        Allocates fractional bitrates {R_l} per layer proportional to sensitivity alpha_l * sigma_l^2
        while strictly adhering to the total parameter bit budget B_total = target_bpw * total_params.
        Uses bisection on the Lagrange multiplier lambda so box constraints [min_bpw, max_bpw]
        and the exact sum constraint sum_l N_l R_l = B_total are simultaneously respected.
        """
        effective_weights = np.maximum(self.sensitivities * self.variances, 1e-12)
        log_w = np.log2(effective_weights)

        # We seek offset C such that sum_l N_l * clip(C + 0.5 * log_w_l, min_bpw, max_bpw) == target_bpw * N_total
        low_c = min_bpw - 0.5 * float(np.max(log_w)) - 10.0
        high_c = max_bpw - 0.5 * float(np.min(log_w)) + 10.0

        allocated_rates = np.full_like(log_w, target_bpw)
        for _ in range(80):
            mid_c = 0.5 * (low_c + high_c)
            cand = np.clip(mid_c + 0.5 * log_w, min_bpw, max_bpw)
            avg_rate = float(np.sum(self.param_counts * cand) / max(self.total_params, 1.0))
            if avg_rate < target_bpw:
                low_c = mid_c
            else:
                high_c = mid_c
            allocated_rates = cand

        return {name: float(rate) for name, rate in zip(self.layer_names, allocated_rates)}


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
