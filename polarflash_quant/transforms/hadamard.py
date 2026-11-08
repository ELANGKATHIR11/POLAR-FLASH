"""
POLAR-FLASH: Orthogonal Fast Walsh-Hadamard Transform & Gaussian Concentration Engine
Implements Part II Sections 2 & 3 of the POLAR-FLASH Mathematical Specification:
1. Fast Walsh-Hadamard Transform (FWHT) with O(d log2 d) complexity.
2. Randomized sign diagonal preprocessing (D_ii in {-1, +1}) for incoherent rotation.
3. Block-wise execution (BlockHadamardTransform) for arbitrary matrix dimensions (e.g. non-power-of-2 SwiGLU layers).
4. Strict energy preservation: ||H w||_2^2 = ||w||_2^2 and invertibility w = H^T z.
5. Gaussian concentration analysis on thin spherical shells: E[||z||^2] = d*sigma^2, Var(||z||^2) = 2*d*sigma^4, R ~ sigma*sqrt(d).
6. Kurtosis, outlier suppression, and distribution diagnostics.
"""

import math
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional


def fast_walsh_hadamard_transform(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Vectorized Fast Walsh-Hadamard Transform (FWHT) along the last dimension.
    The last dimension d must be a power of 2.
    Vectorized using batched tensor reshaping with zero per-element Python loops:
    O(d log2(d)) operations executed in C/CUDA memory-bandwidth speed.
    """
    orig_shape = x.shape
    d = orig_shape[-1]
    if (d & (d - 1)) != 0:
        raise ValueError(f"Last dimension d must be a power of 2, got {d}")

    # Reshape to 2D for batched processing
    out = x.reshape(-1, d).clone()
    h = 1
    while h < d:
        # Reshape to isolate butterfly pairs at distance h
        out = out.view(-1, d // (2 * h), 2, h)
        u = out[:, :, 0, :].clone()
        v = out[:, :, 1, :].clone()
        out[:, :, 0, :] = u + v
        out[:, :, 1, :] = u - v
        out = out.view(-1, d)
        h *= 2

    if normalize:
        out = out / math.sqrt(d)

    return out.reshape(orig_shape)


class RandomizedHadamardTransform:
    """
    Applies randomized orthogonal rotation:
    z = H * D * w
    where D is a diagonal matrix of uniform random signs in {-1, +1}, and H is normalized FWHT.
    Preserves exact Euclidean norm ||z||_2 = ||w||_2 while dispersing outlier coordinate spikes.
    Supports CPU and CUDA execution.
    """
    def __init__(self, dim: int, seed: int = 42, device: str = "cpu"):
        if (dim & (dim - 1)) != 0:
            raise ValueError(f"Dimension must be a power of 2, got {dim}. Use BlockHadamardTransform for arbitrary dimensions.")
        self.dim = dim
        self.device = device

        gen = torch.Generator().manual_seed(seed)
        # Uniform Rademacher signs: {-1, +1}
        self.signs = (torch.randint(0, 2, (dim,), generator=gen, device=device).float() * 2.0 - 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward rotation: z = H * (x * signs)"""
        x_signed = x * self.signs.to(x.device)
        return fast_walsh_hadamard_transform(x_signed, normalize=True)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse rotation: x = (H * z) * signs (since H^T = H and D^-1 = D)"""
        h_inv = fast_walsh_hadamard_transform(z, normalize=True)
        return h_inv * self.signs.to(z.device)


class BlockHadamardTransform:
    """
    Block-wise Randomized Hadamard Transform supporting arbitrary matrix dimensions.
    Partitions the last dimension into subvectors of size block_size (power of 2, default 256 or 512).
    Remaining elements smaller than block_size are processed with the largest power-of-2 divisor or left unrotated.
    Guarantees exact energy preservation ||H w||_2^2 = ||w||_2^2 for any matrix shape.
    """
    def __init__(self, in_features: Optional[int] = None, block_size: int = 256, seed: int = 42, device: str = "cpu"):
        self.block_size_req = block_size
        self.seed = seed
        self.device = device
        self._initialized = False
        if in_features is not None:
            self._init_dims(in_features)

    def _init_dims(self, in_features: int):
        self.in_features = in_features
        # Choose block size as power of 2 <= in_features
        bs = 1
        while bs * 2 <= min(self.block_size_req, in_features) and (bs * 2 & (bs * 2 - 1)) == 0:
            bs *= 2
        self.block_size = max(bs, 2)
        self.num_full_blocks = in_features // self.block_size
        self.remainder = in_features % self.block_size

        # Generate signs for full block
        gen = torch.Generator().manual_seed(self.seed)
        self.signs = (torch.randint(0, 2, (self.block_size,), generator=gen, device=self.device).float() * 2.0 - 1.0)

        # Precompute remainder power-of-2 blocks if remainder > 1
        self.rem_blocks = []
        rem_curr = self.remainder
        while rem_curr > 1:
            p2 = 1
            while p2 * 2 <= rem_curr:
                p2 *= 2
            rem_signs = (torch.randint(0, 2, (p2,), generator=gen, device=self.device).float() * 2.0 - 1.0)
            self.rem_blocks.append((p2, rem_signs))
            rem_curr -= p2
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies block-wise randomized Hadamard rotation."""
        orig_shape = x.shape
        d = orig_shape[-1]
        if not self._initialized or self.in_features != d:
            self._init_dims(d)
        flat_x = x.reshape(-1, d)
        out = flat_x.clone()

        # Transform full blocks
        block_span = self.num_full_blocks * self.block_size
        if block_span > 0:
            full_blocks = flat_x[:, :block_span].reshape(-1, self.num_full_blocks, self.block_size)
            full_blocks_signed = full_blocks * self.signs.to(x.device).unsqueeze(0).unsqueeze(0)
            rotated_blocks = fast_walsh_hadamard_transform(full_blocks_signed, normalize=True)
            out[:, :block_span] = rotated_blocks.reshape(-1, block_span)

        # Transform remainder with cascade of smaller power-of-2 blocks
        offset = block_span
        for p2, rem_signs in self.rem_blocks:
            chunk = flat_x[:, offset:offset + p2]
            chunk_signed = chunk * rem_signs.to(x.device).unsqueeze(0)
            out[:, offset:offset + p2] = fast_walsh_hadamard_transform(chunk_signed, normalize=True)
            offset += p2

        return out.reshape(orig_shape)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverts block-wise randomized Hadamard rotation."""
        orig_shape = z.shape
        d = orig_shape[-1]
        if not self._initialized or self.in_features != d:
            self._init_dims(d)
        flat_z = z.reshape(-1, d)
        out = flat_z.clone()

        block_span = self.num_full_blocks * self.block_size
        if block_span > 0:
            full_blocks = flat_z[:, :block_span].reshape(-1, self.num_full_blocks, self.block_size)
            unrotated = fast_walsh_hadamard_transform(full_blocks, normalize=True)
            unrotated_signed = unrotated * self.signs.to(z.device).unsqueeze(0).unsqueeze(0)
            out[:, :block_span] = unrotated_signed.reshape(-1, block_span)

        offset = block_span
        for p2, rem_signs in self.rem_blocks:
            chunk = flat_z[:, offset:offset + p2]
            unrot_chunk = fast_walsh_hadamard_transform(chunk, normalize=True)
            out[:, offset:offset + p2] = unrot_chunk * rem_signs.to(z.device).unsqueeze(0)
            offset += p2

        return out.reshape(orig_shape)


def verify_gaussian_concentration(
    z_transformed: torch.Tensor,
    dimension: int,
    expected_sigma_sq: Optional[float] = None
) -> Dict[str, Any]:
    """
    Numerically validates Gaussian concentration of transformed vectors on thin spherical shells:
    1. E[||z||_2^2] = d * sigma^2
    2. Var(||z||_2^2) = 2 * d * sigma^4
    3. Radial shell radius: R ~ sigma * sqrt(d)
    4. Relative shell thickness: Delta R / R ~ 1 / sqrt(2 * d)
    """
    flat = z_transformed.detach().cpu().float().view(-1, dimension)
    num_vectors = flat.shape[0]
    if num_vectors < 2:
        raise ValueError(f"Need at least 2 vectors to evaluate concentration variance, got {num_vectors}")

    # Norm squared of each d-dimensional vector: ||z||_2^2
    norms_sq = torch.sum(flat ** 2, dim=-1)
    radii = torch.sqrt(torch.clamp(norms_sq, min=1e-15))

    empirical_e_norm_sq = float(norms_sq.mean().item())
    empirical_var_norm_sq = float(norms_sq.var().item())
    empirical_mean_radius = float(radii.mean().item())
    empirical_std_radius = float(radii.std().item())

    # Theoretical sigma estimation (if not provided)
    sigma_sq = expected_sigma_sq if expected_sigma_sq is not None else (empirical_e_norm_sq / float(dimension))
    theoretical_e_norm_sq = dimension * sigma_sq
    theoretical_var_norm_sq = 2.0 * dimension * (sigma_sq ** 2)
    theoretical_mean_radius = math.sqrt(dimension * sigma_sq)
    theoretical_relative_thickness = 1.0 / math.sqrt(2.0 * dimension)

    error_e = abs(empirical_e_norm_sq - theoretical_e_norm_sq) / max(theoretical_e_norm_sq, 1e-12)
    error_var = abs(empirical_var_norm_sq - theoretical_var_norm_sq) / max(theoretical_var_norm_sq, 1e-12)

    return {
        "dimension": dimension,
        "num_vectors_evaluated": num_vectors,
        "sigma_sq": sigma_sq,
        "empirical_E_norm_sq": empirical_e_norm_sq,
        "theoretical_E_norm_sq": theoretical_e_norm_sq,
        "error_E_norm_sq": error_e,
        "empirical_Var_norm_sq": empirical_var_norm_sq,
        "theoretical_Var_norm_sq": theoretical_var_norm_sq,
        "error_Var_norm_sq": error_var,
        "empirical_mean_radius_R": empirical_mean_radius,
        "theoretical_mean_radius_R": theoretical_mean_radius,
        "empirical_relative_shell_thickness": empirical_std_radius / max(empirical_mean_radius, 1e-12),
        "theoretical_relative_shell_thickness": theoretical_relative_thickness,
        "concentrated_on_shell": error_e < 0.05 and error_var < 0.20
    }


def compute_distribution_diagnostics(w_orig: torch.Tensor, w_rot: torch.Tensor) -> Dict[str, Any]:
    """
    Evaluates Kurtosis, Outlier Frequency, Energy Preservation, and Radial Concentration
    before and after Hadamard rotation.
    """
    orig_flat = w_orig.detach().cpu().float().view(-1)
    rot_flat = w_rot.detach().cpu().float().view(-1)

    # 1. Energy Preservation Check: ||w||^2 vs ||z||^2
    energy_orig = torch.sum(orig_flat ** 2).item()
    energy_rot = torch.sum(rot_flat ** 2).item()
    energy_rel_error = abs(energy_orig - energy_rot) / max(energy_orig, 1e-12)

    # 2. Kurtosis: E[(X - mu)^4] / sigma^4 (Gaussian kurtosis = 3.0)
    def calc_kurtosis(t: torch.Tensor) -> float:
        mu = t.mean()
        std = t.std()
        if std == 0:
            return 0.0
        return float(torch.mean(((t - mu) / std) ** 4).item())

    # 3. Outlier frequency: Fraction of weights with |w| > 3.0 * sigma
    def outlier_rate(t: torch.Tensor) -> float:
        std = t.std()
        mu = t.mean()
        outliers = (torch.abs(t - mu) > 3.0 * std).float().mean()
        return float(outliers.item())

    # 4. Maximum absolute amplitude (peak outlier suppression)
    max_orig = float(torch.max(torch.abs(orig_flat)).item())
    max_rot = float(torch.max(torch.abs(rot_flat)).item())

    return {
        "energy_preservation_relative_error": energy_rel_error,
        "energy_preserved": energy_rel_error < 1e-4,
        "kurtosis_before": calc_kurtosis(orig_flat),
        "kurtosis_after": calc_kurtosis(rot_flat),
        "outlier_rate_before": outlier_rate(orig_flat),
        "outlier_rate_after": outlier_rate(rot_flat),
        "max_outlier_amplitude_before": max_orig,
        "max_outlier_amplitude_after": max_rot,
        "outlier_suppression_ratio": max_orig / max(max_rot, 1e-12)
    }
